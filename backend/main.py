"""
SocialLens Backend — FastAPI + WebSocket
"""

import asyncio
import base64
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
from face_db import FaceDatabase
from social_metrics import assign_social_scores, social_ranking_ids, compute_pairwise_matrix
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
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

cv_pipeline = CVPipeline()
vlm_analyzer = VLMAnalyzer(api_key=ANTHROPIC_API_KEY)
tts_client = ElevenLabsClient(api_key=ELEVENLABS_API_KEY)
face_db = FaceDatabase()

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


def _attention_edges(persons: list, img_w: int, img_h: int) -> dict[str, str]:
    """Map src person_id -> dst person_id if attention vector points to target."""
    centers = {p.person_id: (p.face_center[0], p.face_center[1]) for p in persons}
    diag = max((img_w ** 2 + img_h ** 2) ** 0.5, 1.0)
    edges: dict[str, str] = {}

    for p in persons:
        lm = p.landmark_data or {}
        # Strict mode: social attention uses head/gaze only.
        # Body focus is visualized separately but not used for role/edge decisions.
        vec = lm.get("head_gaze_vector")
        if not vec:
            continue
        vx, vy = float(vec[0]), float(vec[1])
        vnorm = (vx * vx + vy * vy) ** 0.5
        if vnorm < 1e-6:
            continue
        vx /= vnorm
        vy /= vnorm

        sx, sy = centers[p.person_id]
        best_target = None
        best_score = -1.0
        for q in persons:
            if q.person_id == p.person_id:
                continue
            tx, ty = centers[q.person_id]
            dx, dy = tx - sx, ty - sy
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < 1e-6:
                continue
            ux, uy = dx / dist, dy / dist
            dot = vx * ux + vy * uy
            if dot < 0.62:
                continue
            # Prefer closer and better aligned targets.
            score = dot - 0.18 * (dist / diag)
            if score > best_score:
                best_score = score
                best_target = q.person_id
        if best_target:
            edges[p.person_id] = best_target
    return edges


def _roles_groups_center_from_attention(persons: list, img_w: int, img_h: int) -> tuple[dict, list[dict], Optional[str]]:
    if not persons:
        return {}, [], None

    edges = _attention_edges(persons, img_w, img_h)
    ids = [p.person_id for p in persons]
    in_deg = {pid: 0 for pid in ids}
    out_deg = {pid: 0 for pid in ids}
    for src, dst in edges.items():
        out_deg[src] += 1
        in_deg[dst] += 1

    # Build conservative graph: attention edges + strong proximity edges.
    adj: dict[str, set[str]] = {pid: set() for pid in ids}
    for src, dst in edges.items():
        adj[src].add(dst)
        adj[dst].add(src)

    pairwise = compute_pairwise_matrix(persons, img_w, img_h) if len(persons) > 1 else {}
    for (a, b), score in pairwise.items():
        if score >= 70:
            adj[a].add(b)
            adj[b].add(a)

    roles: dict = {}
    for p in persons:
        pid = p.person_id
        indeg = in_deg[pid]
        outdeg = out_deg[pid]
        if indeg >= 2:
            role = "leader"
            conf = min(95, 70 + indeg * 8)
            reason = f"receives attention from {indeg} people"
        elif outdeg >= 1 and indeg >= 1:
            role = "connector"
            conf = 76
            reason = "both attends to others and is attended by others"
        elif outdeg >= 1:
            role = "listener"
            conf = 72
            reason = "attention vector points to another person"
        else:
            role = "observer"
            conf = 65
            reason = "no strong directed attention evidence"
        roles[pid] = {"role": role, "confidence": conf, "reasoning": reason}

    center = None
    if len(persons) > 1:
        ranked = sorted(persons, key=lambda p: (in_deg[p.person_id], p.social_engagement_score), reverse=True)
        top = ranked[0]
        if in_deg[top.person_id] >= 2:
            center = top.person_id

    # Connected components -> groups
    visited: set[str] = set()
    groups: list[dict] = []
    gid = 1
    for pid in ids:
        if pid in visited:
            continue
        stack = [pid]
        comp: list[str] = []
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            comp.append(cur)
            for nxt in adj[cur]:
                if nxt not in visited:
                    stack.append(nxt)
        if len(comp) < 2:
            continue
        members = [p for p in persons if p.person_id in set(comp)]
        x1 = min(p.bbox[0] for p in members)
        y1 = min(p.bbox[1] for p in members)
        x2 = max(p.bbox[0] + p.bbox[2] for p in members)
        y2 = max(p.bbox[1] + p.bbox[3] for p in members)
        cohesion_edges = 0
        for m in comp:
            cohesion_edges += len(adj[m])
        cohesion = min(95, max(40, int(45 + cohesion_edges * 6)))
        groups.append(
            {
                "group_id": f"G{gid}",
                "member_ids": sorted(comp, key=lambda x: int(x[1:]) if x[1:].isdigit() else x),
                "group_type": "conversation" if cohesion >= 70 else "casual",
                "cohesion_score": cohesion,
                "bounding_region": {"x": int(x1), "y": int(y1), "w": int(x2 - x1), "h": int(y2 - y1)},
            }
        )
        gid += 1

    return roles, groups, center


async def run_full_pipeline(
    image_bgr: np.ndarray,
    *,
    run_vlm: bool = True,
    sticky_vlm: Optional[dict] = None,
) -> dict:
    """CV → social metrics → VLM (async). Returns full analysis dict."""
    t0 = time.monotonic()
    img_h, img_w = image_bgr.shape[:2]

    # CV pipeline (synchronous, fast)
    persons = cv_pipeline.process(image_bgr)

    # Face recognition: identify known people by name
    for p in persons:
        matched_name = face_db.identify(image_bgr, tuple(p.bbox))
        if matched_name:
            p.name = matched_name

    persons = assign_social_scores(persons, img_w, img_h)

    person_dicts = [p.to_dict() for p in persons]
    person_bboxes = [{"person_id": p.person_id, "bbox": p.bbox} for p in persons]

    # VLM analysis (async, optional for low-latency live mode)
    vlm_result = {}
    if run_vlm and persons:
        vlm_result = await vlm_analyzer.analyze_scene(image_bgr, person_bboxes)
    elif sticky_vlm:
        # Reuse first-frame VLM narrative for subsequent live frames.
        vlm_result = {
            "dynamics_summary": sticky_vlm.get("dynamics_summary", ""),
            "interesting_observations": sticky_vlm.get("interesting_observations", []),
        }

    # Annotated image
    annotated = draw_annotations(image_bgr, persons)
    annotated_b64 = encode_image_b64(annotated)

    elapsed_ms = round((time.monotonic() - t0) * 1000)

    roles, groups, center_from_attention = _roles_groups_center_from_attention(persons, img_w, img_h)
    social_center = center_from_attention

    return {
        "image_id": str(uuid.uuid4()),
        "persons": person_dicts,
        "groups": groups,
        "roles": roles,
        "social_center": social_center,
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


# ---------------------------------------------------------------------------
# Personal Face Database endpoints
# ---------------------------------------------------------------------------

@app.get("/api/face-db")
async def face_db_list():
    """List all people registered in the face database."""
    return {"persons": face_db.list_persons()}


@app.post("/api/face-db")
async def face_db_add(name: str, file: UploadFile = File(...)):
    """Register a person. Send their name and a clear face photo."""
    raw = await file.read()
    try:
        image_bgr = decode_image(raw)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    result_code = face_db.add_person(name.strip(), image_bgr)
    if result_code == "not_installed":
        raise HTTPException(
            status_code=503,
            detail=(
                "face_recognition library is not installed on the server. "
                "Fix: sudo apt install cmake build-essential && pip install face-recognition"
            ),
        )
    if result_code == "no_face":
        raise HTTPException(
            status_code=422,
            detail="No face detected in the uploaded image. Please use a clear, well-lit front-facing photo.",
        )
    return {"status": "registered", "name": name.strip(), "total": len(face_db.list_persons())}


@app.delete("/api/face-db/{name}")
async def face_db_remove(name: str):
    """Remove a person from the face database."""
    removed = face_db.remove_person(name)
    if not removed:
        raise HTTPException(status_code=404, detail=f"'{name}' not found in face database.")
    return {"status": "removed", "name": name}


@app.post("/api/dominance/{image_id}/{person_id_a}/{person_id_b}")
async def analyze_dominance(image_id: str, person_id_a: str, person_id_b: str):
    """Analyze perceived dominance between two specific people in a cached image."""
    if image_id not in image_store:
        raise HTTPException(status_code=404, detail="Image not found. Re-analyze first.")

    image_bgr, analysis = image_store[image_id]

    person_a = next((p for p in analysis["persons"] if p["person_id"] == person_id_a), None)
    person_b = next((p for p in analysis["persons"] if p["person_id"] == person_id_b), None)

    if person_a is None:
        raise HTTPException(status_code=404, detail=f"Person {person_id_a} not found.")
    if person_b is None:
        raise HTTPException(status_code=404, detail=f"Person {person_id_b} not found.")
    if person_id_a == person_id_b:
        raise HTTPException(status_code=400, detail="Must select two different people.")

    result = await vlm_analyzer.analyze_dominance(image_bgr, person_a, person_b)
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
    first_frame = True
    sticky_vlm: dict = {}
    try:
        while True:
            # Expect binary frame (JPEG bytes) from client
            data = await websocket.receive_bytes()
            try:
                image_bgr = decode_image(data)
            except ValueError:
                await websocket.send_json({"error": "Invalid image data"})
                continue

            # First live frame: full pipeline (with VLM).
            # Following frames: CV-only for low latency.
            result = await run_full_pipeline(
                image_bgr,
                run_vlm=first_frame,
                sticky_vlm=sticky_vlm,
            )
            if first_frame:
                sticky_vlm = {
                    "dynamics_summary": result.get("dynamics_summary", ""),
                    "interesting_observations": result.get("interesting_observations", []),
                }
                first_frame = False

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
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
