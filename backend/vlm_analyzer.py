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


# ---------------------------------------------------------------------------
# Fallback when API fails
# ---------------------------------------------------------------------------

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
