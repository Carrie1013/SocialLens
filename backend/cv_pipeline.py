"""
CV Pipeline: MediaPipe Tasks API (Python 3.13 compatible).
- Primary: MediaPipe FaceDetector (full-range model) + FaceLandmarker + PoseLandmarker
- Fallback: OpenCV Haar cascade when MediaPipe detects 0 faces
Model files are downloaded automatically on first run.
"""

import math
import logging
import pathlib
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model file management
# ---------------------------------------------------------------------------

MODEL_DIR = pathlib.Path(__file__).parent / "models"

MODELS = {
    "face_detector": (
        "blaze_face_short_range.tflite",
        "https://storage.googleapis.com/mediapipe-models/face_detector/"
        "blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
    ),
    "face_landmarker": (
        "face_landmarker.task",
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task",
    ),
    "pose_landmarker": (
        "pose_landmarker_lite.task",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    ),
}


def ensure_models() -> dict[str, pathlib.Path]:
    """Download missing model files and return name→path mapping."""
    MODEL_DIR.mkdir(exist_ok=True)
    paths: dict[str, pathlib.Path] = {}
    for name, (filename, url) in MODELS.items():
        path = MODEL_DIR / filename
        paths[name] = path
        if not path.exists():
            logger.info("Downloading %s model…", name)
            try:
                urllib.request.urlretrieve(url, path)
                logger.info("  → saved %s (%.1f MB)", filename, path.stat().st_size / 1e6)
            except Exception as e:
                logger.error("Failed to download %s: %s", name, e)
                raise RuntimeError(f"Could not download model '{name}'.") from e
        else:
            logger.debug("Model already cached: %s", filename)
    return paths


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PersonData:
    person_id: str
    bbox: list[int]           # [x, y, w, h]  pixel coords
    estimated_depth: float
    body_orientation: str
    expression: str
    face_area_px: int
    social_engagement_score: float = 0.0
    social_rank: int = 0
    face_center: tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))
    landmark_data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "person_id": self.person_id,
            "bbox": self.bbox,
            "estimated_depth": round(self.estimated_depth, 2),
            "body_orientation": self.body_orientation,
            "expression": self.expression,
            "face_area_px": self.face_area_px,
            "social_engagement_score": round(self.social_engagement_score, 1),
            "social_rank": self.social_rank,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_dist(p1, p2) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def _resize_for_detection(img: np.ndarray, max_dim: int = 1280) -> tuple[np.ndarray, float]:
    """Resize large images for faster/more reliable detection. Returns (img, scale)."""
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img, 1.0
    scale = max_dim / max(h, w)
    resized = cv2.resize(img, (int(w * scale), int(h * scale)))
    return resized, scale


# ---------------------------------------------------------------------------
# Depth estimation
# ---------------------------------------------------------------------------

REFERENCE_FACE_AREA = 15_000  # px² baseline

def estimate_depth(face_area_px: int) -> float:
    if face_area_px <= 0:
        return 10.0
    depth = math.sqrt(REFERENCE_FACE_AREA / face_area_px) * 3.0
    return round(min(max(depth, 0.1), 10.0), 2)


# ---------------------------------------------------------------------------
# Body orientation
# ---------------------------------------------------------------------------

def classify_orientation(pose_landmarks: list | None, img_w: int, img_h: int) -> str:
    if not pose_landmarks or len(pose_landmarks) < 13:
        return "UNKNOWN"
    ls = pose_landmarks[11]
    rs = pose_landmarks[12]
    vis_ls = getattr(ls, "visibility", None) or 0.0
    vis_rs = getattr(rs, "visibility", None) or 0.0
    if vis_ls < 0.25 or vis_rs < 0.25:
        return "UNKNOWN"
    dx = (rs.x - ls.x) * img_w
    dy = (rs.y - ls.y) * img_h
    shoulder_width_px = math.sqrt(dx * dx + dy * dy)
    dz = rs.z - ls.z
    angle_deg = math.degrees(math.atan2(dy, dx))
    if abs(dz) > 0.15:
        return "PROFILE_LEFT" if dz > 0 else "PROFILE_RIGHT"
    if shoulder_width_px < 20:
        return "TURNED_AWAY"
    if abs(angle_deg) < 20:
        return "FACING_CAMERA"
    return "ANGLED"


# ---------------------------------------------------------------------------
# Facial expression
# ---------------------------------------------------------------------------

UPPER_LIP = 13;  LOWER_LIP = 14
LEFT_MOUTH_CORNER = 61;  RIGHT_MOUTH_CORNER = 291
LEFT_EYE_TOP = 159;  LEFT_EYE_BOTTOM = 145
RIGHT_EYE_TOP = 386;  RIGHT_EYE_BOTTOM = 374
NOSE_TIP = 1
LEFT_EYE_INNER = 133;  RIGHT_EYE_INNER = 362


def classify_expression(face_landmarks: list, img_w: int, img_h: int) -> str:
    if not face_landmarks or len(face_landmarks) < 400:
        return "NEUTRAL"

    def pt(idx):
        lm = face_landmarks[idx]
        return (lm.x * img_w, lm.y * img_h)

    mouth_open  = _safe_dist(pt(UPPER_LIP), pt(LOWER_LIP))
    mouth_width = _safe_dist(pt(LEFT_MOUTH_CORNER), pt(RIGHT_MOUTH_CORNER))
    mouth_ratio = mouth_open / (mouth_width + 1e-6)

    left_eye_open  = _safe_dist(pt(LEFT_EYE_TOP),  pt(LEFT_EYE_BOTTOM))
    right_eye_open = _safe_dist(pt(RIGHT_EYE_TOP), pt(RIGHT_EYE_BOTTOM))
    eye_open_avg   = (left_eye_open + right_eye_open) / 2.0
    eye_width_ref  = _safe_dist(pt(LEFT_EYE_INNER), pt(RIGHT_EYE_INNER))
    eye_ratio      = eye_open_avg / (eye_width_ref + 1e-6)

    left_corner_y  = pt(LEFT_MOUTH_CORNER)[1]
    right_corner_y = pt(RIGHT_MOUTH_CORNER)[1]
    upper_lip_y    = pt(UPPER_LIP)[1]
    corner_lift    = upper_lip_y - (left_corner_y + right_corner_y) / 2.0
    smile_ratio    = corner_lift / (mouth_width + 1e-6)

    if mouth_ratio > 0.35 and smile_ratio > 0.05:
        return "LAUGHING"
    if mouth_ratio > 0.25:
        return "TALKING"
    if eye_ratio > 0.25 and mouth_ratio > 0.15:
        return "SURPRISED"
    if smile_ratio > 0.06:
        return "SMILING"
    if eye_ratio < 0.08:
        return "FOCUSED"
    return "NEUTRAL"


def _find_closest_face_landmarks(all_faces: list[list], norm_cx: float, norm_cy: float) -> list | None:
    best, best_dist = None, float("inf")
    for face_lms in all_faces:
        if not face_lms or len(face_lms) <= NOSE_TIP:
            continue
        nose = face_lms[NOSE_TIP]
        d = _safe_dist((nose.x, nose.y), (norm_cx, norm_cy))
        if d < best_dist:
            best_dist = d
            best = face_lms
    return best


# ---------------------------------------------------------------------------
# OpenCV Haar cascade fallback
# ---------------------------------------------------------------------------

def _haar_detect_faces(image_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Fallback face detection using OpenCV Haar cascade."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    if len(faces) == 0:
        return []
    return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]


# ---------------------------------------------------------------------------
# Main CV Pipeline
# ---------------------------------------------------------------------------

class CVPipeline:
    def __init__(self):
        try:
            model_paths = ensure_models()
        except RuntimeError as e:
            logger.warning("Model download failed (%s) — using Haar-only mode", e)
            model_paths = {}

        self._face_detector = None
        self._face_landmarker = None
        self._pose_landmarker = None

        if model_paths.get("face_detector"):
            try:
                det_opts = mp_vision.FaceDetectorOptions(
                    base_options=mp_python.BaseOptions(
                        model_asset_path=str(model_paths["face_detector"])
                    ),
                    min_detection_confidence=0.2,
                )
                self._face_detector = mp_vision.FaceDetector.create_from_options(det_opts)
                logger.info("MediaPipe FaceDetector ready")
            except Exception as e:
                logger.warning("FaceDetector init failed: %s", e)

        if model_paths.get("face_landmarker"):
            try:
                lm_opts = mp_vision.FaceLandmarkerOptions(
                    base_options=mp_python.BaseOptions(
                        model_asset_path=str(model_paths["face_landmarker"])
                    ),
                    num_faces=8,
                    min_face_detection_confidence=0.2,
                    min_face_presence_confidence=0.2,
                    min_tracking_confidence=0.2,
                )
                self._face_landmarker = mp_vision.FaceLandmarker.create_from_options(lm_opts)
                logger.info("MediaPipe FaceLandmarker ready")
            except Exception as e:
                logger.warning("FaceLandmarker init failed: %s", e)

        if model_paths.get("pose_landmarker"):
            try:
                pose_opts = mp_vision.PoseLandmarkerOptions(
                    base_options=mp_python.BaseOptions(
                        model_asset_path=str(model_paths["pose_landmarker"])
                    ),
                    min_pose_detection_confidence=0.3,
                    min_pose_presence_confidence=0.3,
                    min_tracking_confidence=0.3,
                )
                self._pose_landmarker = mp_vision.PoseLandmarker.create_from_options(pose_opts)
                logger.info("MediaPipe PoseLandmarker ready")
            except Exception as e:
                logger.warning("PoseLandmarker init failed: %s", e)

        logger.info("CVPipeline ready (MP face=%s, landmarks=%s, pose=%s, Haar=always-on)",
                    self._face_detector is not None,
                    self._face_landmarker is not None,
                    self._pose_landmarker is not None)

    def process(self, image_bgr: np.ndarray) -> list[PersonData]:
        img_h, img_w = image_bgr.shape[:2]
        logger.info("Processing image %dx%d", img_w, img_h)

        # Resize large images for more reliable detection
        work_img, det_scale = _resize_for_detection(image_bgr, max_dim=1280)
        work_h, work_w = work_img.shape[:2]

        image_rgb = cv2.cvtColor(work_img, cv2.COLOR_BGR2RGB)

        # --- Face detection: MediaPipe first, then Haar fallback ---
        bboxes: list[tuple[int, int, int, int]] = []
        used_haar = False

        if self._face_detector is not None:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            det_result = self._face_detector.detect(mp_image)
            mp_detections = det_result.detections or []
            logger.info("MediaPipe detected %d face(s)", len(mp_detections))
            for d in mp_detections:
                bb = d.bounding_box
                x = max(0, bb.origin_x)
                y = max(0, bb.origin_y)
                w = min(bb.width,  work_w - x)
                h = min(bb.height, work_h - y)
                if w > 0 and h > 0:
                    bboxes.append((x, y, w, h))

        if not bboxes:
            logger.info("Trying Haar cascade fallback…")
            haar_bboxes = _haar_detect_faces(work_img)
            if haar_bboxes:
                bboxes = haar_bboxes
                used_haar = True
                logger.info("Haar found %d face(s)", len(bboxes))
            else:
                logger.info("No faces detected in this image")
                return []

        # --- Face landmarks (MediaPipe) ---
        all_face_lms: list[list] = []
        if self._face_landmarker is not None:
            if not hasattr(self, "_mp_image_cache") or self._mp_image_cache is None:
                pass
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            lm_result = self._face_landmarker.detect(mp_image)
            all_face_lms = lm_result.face_landmarks or []

        # --- Pose ---
        pose_lms = None
        if self._pose_landmarker is not None and not used_haar:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            pose_result = self._pose_landmarker.detect(mp_image)
            pose_lms = pose_result.pose_landmarks[0] if pose_result.pose_landmarks else None

        # --- Build PersonData ---
        persons: list[PersonData] = []
        for idx, (x, y, w, h) in enumerate(bboxes):
            person_id = f"P{idx + 1}"

            orig_x = int(x / det_scale)
            orig_y = int(y / det_scale)
            orig_w = int(w / det_scale)
            orig_h = int(h / det_scale)
            face_area = max(orig_w * orig_h, 1)

            face_cx = (x + w / 2) / work_w
            face_cy = (y + h / 2) / work_h

            depth       = estimate_depth(face_area)
            orientation = classify_orientation(pose_lms, work_w, work_h)

            expression = "NEUTRAL"
            if all_face_lms:
                closest = _find_closest_face_landmarks(all_face_lms, face_cx, face_cy)
                if closest:
                    expression = classify_expression(closest, work_w, work_h)

            persons.append(PersonData(
                person_id=person_id,
                bbox=[orig_x, orig_y, orig_w, orig_h],
                estimated_depth=depth,
                body_orientation=orientation,
                expression=expression,
                face_area_px=face_area,
                face_center=(orig_x + orig_w / 2, orig_y + orig_h / 2),
            ))

        logger.info("Returning %d person(s): %s", len(persons), [p.person_id for p in persons])
        return persons

    def close(self):
        if self._face_detector:
            self._face_detector.close()
        if self._face_landmarker:
            self._face_landmarker.close()
        if self._pose_landmarker:
            self._pose_landmarker.close()


# ---------------------------------------------------------------------------
# Annotated image rendering
# ---------------------------------------------------------------------------

EXPRESSION_EMOJI = {
    "NEUTRAL":   "",
    "SMILING":   ":)",
    "TALKING":   "...",
    "SURPRISED": "O_O",
    "FOCUSED":   ">_<",
    "LAUGHING":  ":D",
    "UNKNOWN":   "?",
}

SCORE_COLORS = {
    "high":   (0, 255, 136),
    "medium": (0, 170, 255),
    "low":    (68, 68, 255),
}


def score_color(score: float) -> tuple[int, int, int]:
    if score >= 60:
        return SCORE_COLORS["high"]
    if score >= 35:
        return SCORE_COLORS["medium"]
    return SCORE_COLORS["low"]


def draw_annotations(image_bgr: np.ndarray, persons: list[PersonData]) -> np.ndarray:
    annotated = image_bgr.copy()
    for p in persons:
        x, y, w, h = p.bbox
        color = score_color(p.social_engagement_score)

        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

        label = (f"{p.person_id} "
                 f"{EXPRESSION_EMOJI.get(p.expression, '')} "
                 f"score:{p.social_engagement_score:.0f}")
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # Clamp label above bbox if near top of image
        label_y = max(y, th + 10)
        cv2.rectangle(annotated, (x, label_y - th - 8), (x + tw + 4, label_y), color, -1)
        cv2.putText(annotated, label, (x + 2, label_y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        depth_label = f"d:{p.estimated_depth:.1f}m"
        cv2.putText(annotated, depth_label, (x + 2, y + h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    return annotated
