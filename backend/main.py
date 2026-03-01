"""
SocialLens Backend — FastAPI + WebSocket
"""

import asyncio
import base64
import io
import json
import logging
import os
import time
import uuid
from typing import Optional

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from cv_pipeline import CVPipeline, draw_annotations
from elevenlabs_client import ElevenLabsClient, select_voice_id
from social_metrics import assign_social_scores, social_ranking_ids
from vlm_analyzer import VLMAnalyzer

# ---------------------------------------------------------------------------
# Logging & env
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")

if not ANTHROPIC_API_KEY:
    logger.warning("ANTHROPIC_API_KEY not set — VLM analysis will use fallback responses.")
if not ELEVENLABS_API_KEY:
    logger.warning("ELEVENLABS_API_KEY not set — TTS will be disabled.")

# ---------------------------------------------------------------------------
# App & services
# ---------------------------------------------------------------------------

app = FastAPI(title="SocialLens API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

cv_pipeline = CVPipeline()
vlm_analyzer = VLMAnalyzer(api_key=ANTHROPIC_API_KEY)
tts_client = ElevenLabsClient(api_key=ELEVENLABS_API_KEY)

# In-memory store: image_id → (image_bgr, analysis_result)
image_store: dict[str, tuple[np.ndarray, dict]] = {}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img


def encode_image_b64(image_bgr: np.ndarray, fmt: str = ".jpg") -> str:
    params = [cv2.IMWRITE_JPEG_QUALITY, 85] if fmt == ".jpg" else []
    _, buf = cv2.imencode(fmt, image_bgr, params)
    return base64.b64encode(buf).decode("utf-8")


async def run_full_pipeline(image_bgr: np.ndarray) -> dict:
    """CV → social metrics → VLM (async). Returns full analysis dict."""
    t0 = time.monotonic()
    img_h, img_w = image_bgr.shape[:2]

    # CV pipeline (synchronous, fast)
    persons = cv_pipeline.process(image_bgr)
    persons = assign_social_scores(persons, img_w, img_h)

    person_dicts = [p.to_dict() for p in persons]
    person_bboxes = [{"person_id": p.person_id, "bbox": p.bbox} for p in persons]

    # VLM analysis (async)
    vlm_result = {}
    if persons:
        vlm_result = await vlm_analyzer.analyze_scene(image_bgr, person_bboxes)

    # Annotated image
    annotated = draw_annotations(image_bgr, persons)
    annotated_b64 = encode_image_b64(annotated)

    elapsed_ms = round((time.monotonic() - t0) * 1000)

    return {
        "image_id": str(uuid.uuid4()),
        "persons": person_dicts,
        "groups": vlm_result.get("groups", []),
        "roles": vlm_result.get("roles", {}),
        "social_center": vlm_result.get("social_center"),
        "dynamics_summary": vlm_result.get("dynamics_summary", ""),
        "interesting_observations": vlm_result.get("interesting_observations", []),
        "social_ranking": social_ranking_ids(persons),
        "annotated_image": annotated_b64,
        "processing_time_ms": elapsed_ms,
    }


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "vlm_enabled": bool(ANTHROPIC_API_KEY),
        "tts_enabled": tts_client.is_available(),
    }


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    """Upload an image, run full CV + VLM pipeline."""
    raw = await file.read()
    try:
        image_bgr = decode_image(raw)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    result = await run_full_pipeline(image_bgr)

    # Cache image + result for voice generation
    image_store[result["image_id"]] = (image_bgr, result)

    return result


@app.post("/api/person/{image_id}/{person_id}/voice")
async def generate_voice(image_id: str, person_id: str):
    """Generate TTS audio for a clicked person."""
    if image_id not in image_store:
        raise HTTPException(status_code=404, detail="Image not found. Re-upload to analyze.")

    image_bgr, analysis = image_store[image_id]
    img_h, img_w = image_bgr.shape[:2]

    # Find person bbox
    person_info = next((p for p in analysis["persons"] if p["person_id"] == person_id), None)
    if person_info is None:
        raise HTTPException(status_code=404, detail=f"Person {person_id} not found.")

    x, y, w, h = person_info["bbox"]
    # 2× crop with padding
    pad = max(w, h) // 4
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(img_w, x + w + pad)
    y2 = min(img_h, y + h + pad)
    crop = image_bgr[y1:y2, x1:x2]

    # Get voice profile from VLM
    profile = await vlm_analyzer.generate_voice_profile(crop)
    dialogue = profile.get("sample_dialogue", "Hello, it's great to meet you.")

    voice_id = select_voice_id(
        profile.get("gender_presentation", "androgynous"),
        profile.get("personality_impression", "calm"),
    )

    audio_bytes = await tts_client.generate_audio_async(dialogue, voice_id)

    if audio_bytes is None:
        raise HTTPException(status_code=503, detail="TTS generation failed or ElevenLabs not configured.")

    return Response(
        content=audio_bytes,
        media_type="audio/mpeg",
        headers={
            "X-Voice-Profile": json.dumps(profile),
            "X-Dialogue": dialogue,
        },
    )


# ---------------------------------------------------------------------------
# WebSocket — Live mode
# ---------------------------------------------------------------------------

@app.websocket("/api/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket client connected")
    try:
        while True:
            # Expect binary frame (JPEG bytes) from client
            data = await websocket.receive_bytes()
            try:
                image_bgr = decode_image(data)
            except ValueError:
                await websocket.send_json({"error": "Invalid image data"})
                continue

            result = await run_full_pipeline(image_bgr)
            image_id = result["image_id"]
            image_store[image_id] = (image_bgr, result)

            await websocket.send_json(result)

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error("WebSocket error: %s", e)
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    logger.info("SocialLens API started.")


@app.on_event("shutdown")
async def shutdown():
    cv_pipeline.close()
    logger.info("SocialLens API shut down.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
