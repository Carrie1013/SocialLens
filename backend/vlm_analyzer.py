"""
VLM Analyzer: Claude API integration for social dynamics analysis.
Includes result caching to minimise API calls.
"""

import asyncio
import base64
import json
import logging
import time
from typing import Optional

import anthropic
import cv2
import numpy as np

logger = logging.getLogger(__name__)

CLAUDE_MODEL = "claude-sonnet-4-6"
CACHE_TTL_SECONDS = 3.0

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

ANALYSIS_PROMPT_TEMPLATE = """Analyze this image of people and provide social dynamics insights in JSON format.

Person bounding boxes are provided as context: {person_bboxes}

Important calibration rules:
- Do NOT assume eye contact or attention unless clearly visible.
- If gaze/body orientation is uncertain, assign role "observer" with lower confidence.
- Use "social_center": null when there is no clearly dominant social focus.
- Avoid placing all people in one fully connected group unless clearly interacting.

Return ONLY valid JSON with this structure (no markdown, no explanation):
{{
  "groups": [
    {{
      "group_id": "G1",
      "member_ids": ["P1", "P2"],
      "group_type": "conversation|presentation|casual",
      "cohesion_score": 75,
      "bounding_region": {{"x": 0, "y": 0, "w": 100, "h": 100}}
    }}
  ],
  "roles": {{
    "P1": {{
      "role": "leader|listener|speaker|observer|connector",
      "confidence": 85,
      "reasoning": "brief explanation"
    }}
  }},
  "social_center": "P1|null",
  "dynamics_summary": "2-3 sentence description of the overall social scene",
  "interesting_observations": ["observation 1", "observation 2", "observation 3"]
}}
"""

DOMINANCE_PROMPT_TEMPLATE = """You are analyzing perceived social dominance between two specific people in this image.

Person A ({id_a}):
- Engagement score: {score_a}/100, Expression: {expression_a}, Orientation: {orientation_a}, Depth: {depth_a}m

Person B ({id_b}):
- Engagement score: {score_b}/100, Expression: {expression_b}, Orientation: {orientation_b}, Depth: {depth_b}m

The image has been annotated: Person A has a RED bounding box with label "A", Person B has a CYAN bounding box with label "B".

Analyze visual dominance cues between these two people only:
- Body expansion and space occupation (who takes up more physical space)
- Gaze direction (who looks at whom, who commands attention)
- Emotional intensity and expressiveness
- Physical positioning, posture, and orientation toward each other
- Overall social presence and confidence signals

Return ONLY valid JSON (no markdown, no explanation):
{{
  "dominant_person": "A",
  "dominance_score_a": 0.72,
  "dominance_score_b": 0.38,
  "engagement_score": 0.65,
  "relationship_dynamic": "one-sided leadership",
  "reasoning": "2-3 sentences explaining the visual cues that indicate dominance.",
  "body_language_a": "brief description of Person A body language",
  "body_language_b": "brief description of Person B body language"
}}

Rules:
- dominant_person must be "A", "B", or "equal"
- dominance_score_a and dominance_score_b are floats 0.0-1.0 (the dominant person's score should be higher)
- engagement_score is 0.0-1.0 (how engaged they are with each other)
- relationship_dynamic is a short phrase (e.g. "mutual engagement", "hierarchical", "parallel attention", "one-sided focus")
"""

VOICE_PROFILE_PROMPT = """Analyze this person's appearance and generate a character voice profile.
Return ONLY valid JSON (no markdown, no explanation):
{
  "age_estimate": "20s|30s|40s|50s+",
  "gender_presentation": "masculine|feminine|androgynous",
  "personality_impression": "confident|shy|energetic|calm|authoritative|playful",
  "accent_suggestion": "american|british|neutral",
  "speaking_style": "fast-paced|measured|enthusiastic|monotone",
  "sample_dialogue": "A natural sentence this person might say based on their expression and context (max 25 words)"
}
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _image_to_b64(image_bgr: np.ndarray, max_dim: int = 1024) -> str:
    """Resize image if needed, then convert to base64 JPEG."""
    h, w = image_bgr.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        image_bgr = cv2.resize(image_bgr, (int(w * scale), int(h * scale)))
    _, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode("utf-8")


def _parse_json_response(text: str) -> dict:
    """Robustly parse JSON from LLM response, stripping markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return json.loads(text)


# ---------------------------------------------------------------------------
# VLM Analyzer
# ---------------------------------------------------------------------------

class VLMAnalyzer:
    def __init__(self, api_key: str):
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self._cache: dict[str, tuple[dict, float]] = {}   # key → (result, timestamp)
        self._lock = asyncio.Lock()

    def _cache_key(self, person_bboxes: list[dict]) -> str:
        """Coarse cache key based on bbox positions (quantised to 20px grid)."""
        quantised = [
            {k: (v // 20 * 20 if isinstance(v, int) else v) for k, v in b.items()}
            for b in person_bboxes
        ]
        return str(sorted(str(q) for q in quantised))

    async def analyze_scene(
        self,
        image_bgr: np.ndarray,
        person_bboxes: list[dict],
    ) -> dict:
        """
        Run scene-level social dynamics analysis.
        Returns the parsed JSON dict from Claude.
        """
        cache_key = self._cache_key(person_bboxes)
        async with self._lock:
            cached = self._cache.get(cache_key)
            if cached and (time.monotonic() - cached[1]) < CACHE_TTL_SECONDS:
                logger.debug("VLM cache hit")
                return cached[0]

        b64 = _image_to_b64(image_bgr)
        prompt = ANALYSIS_PROMPT_TEMPLATE.format(person_bboxes=json.dumps(person_bboxes))

        try:
            response = await self.client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": b64,
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )
            result = _parse_json_response(response.content[0].text)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse VLM JSON: %s", e)
            result = _empty_analysis(person_bboxes)
        except Exception as e:
            logger.error("VLM API error: %s", e)
            result = _empty_analysis(person_bboxes)

        async with self._lock:
            self._cache[cache_key] = (result, time.monotonic())

        return result

    async def generate_voice_profile(
        self,
        person_crop_bgr: np.ndarray,
    ) -> dict:
        """
        Generate a voice/character profile for a single person crop.
        Returns parsed JSON dict.
        """
        b64 = _image_to_b64(person_crop_bgr, max_dim=512)

        try:
            response = await self.client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=512,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": b64,
                                },
                            },
                            {"type": "text", "text": VOICE_PROFILE_PROMPT},
                        ],
                    }
                ],
            )
            return _parse_json_response(response.content[0].text)
        except Exception as e:
            logger.error("Voice profile VLM error: %s", e)
            return {
                "age_estimate": "30s",
                "gender_presentation": "androgynous",
                "personality_impression": "calm",
                "accent_suggestion": "neutral",
                "speaking_style": "measured",
                "sample_dialogue": "Hello, it's nice to meet you here today.",
            }


    async def analyze_dominance(
        self,
        image_bgr: np.ndarray,
        person_a: dict,
        person_b: dict,
    ) -> dict:
        """
        Analyze perceived dominance between two specific people.
        Draws colored A/B bounding boxes on image before sending to Claude.
        """
        annotated = image_bgr.copy()
        for person, color, label in [
            (person_a, (0, 0, 220), "A"),    # red for A
            (person_b, (220, 220, 0), "B"),  # cyan for B
        ]:
            x, y, w, h = person["bbox"]
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 3)
            cv2.putText(annotated, label, (x + 6, y + 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)

        b64 = _image_to_b64(annotated)
        prompt = DOMINANCE_PROMPT_TEMPLATE.format(
            id_a=person_a.get("person_id", "A"),
            score_a=person_a.get("social_engagement_score", 50),
            expression_a=person_a.get("expression", "UNKNOWN"),
            orientation_a=person_a.get("body_orientation", "UNKNOWN"),
            depth_a=person_a.get("estimated_depth", "?"),
            id_b=person_b.get("person_id", "B"),
            score_b=person_b.get("social_engagement_score", 50),
            expression_b=person_b.get("expression", "UNKNOWN"),
            orientation_b=person_b.get("body_orientation", "UNKNOWN"),
            depth_b=person_b.get("estimated_depth", "?"),
        )

        try:
            response = await self.client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=512,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": b64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }],
            )
            return _parse_json_response(response.content[0].text)
        except json.JSONDecodeError as e:
            logger.error("Dominance parse error: %s", e)
            return _empty_dominance()
        except Exception as e:
            logger.error("Dominance VLM error: %s", e)
            return _empty_dominance()


# ---------------------------------------------------------------------------
# Fallback when API fails
# ---------------------------------------------------------------------------

def _empty_dominance() -> dict:
    return {
        "dominant_person": "equal",
        "dominance_score_a": 0.5,
        "dominance_score_b": 0.5,
        "engagement_score": 0.5,
        "relationship_dynamic": "analysis unavailable",
        "reasoning": "Could not analyze dominance dynamics at this time.",
        "body_language_a": "unknown",
        "body_language_b": "unknown",
    }


def _empty_analysis(person_bboxes: list[dict]) -> dict:
    ids = [b.get("person_id", f"P{i+1}") for i, b in enumerate(person_bboxes)]
    groups = []
    if len(ids) >= 2:
        groups = [{"group_id": "G1", "member_ids": ids, "group_type": "casual",
                   "cohesion_score": 50, "bounding_region": {"x": 0, "y": 0, "w": 640, "h": 480}}]
    roles = {pid: {"role": "observer", "confidence": 50, "reasoning": "Default fallback"} for pid in ids}
    return {
        "groups": groups,
        "roles": roles,
        "social_center": None,
        "dynamics_summary": "Analysis temporarily unavailable.",
        "interesting_observations": [],
    }
