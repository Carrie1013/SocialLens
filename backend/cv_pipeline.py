"""
CV Pipeline: mature human detection with YOLO (person class).

Primary detector:
- Ultralytics YOLO (class 0 = person)

Fallback detector:
- OpenCV HOG people detector

Notes:
- If YOLO weights are missing, Ultralytics will download them on first run.
- Configure model via env var: YOLO_MODEL (default: yolov8n.pt)
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from head_gaze_models import HeadGazeEstimator

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PersonData:
    person_id: str
    bbox: list[int]           # [x, y, w, h] pixel coords
    estimated_depth: float
    body_orientation: str
    expression: str
    face_area_px: int
    social_engagement_score: float = 0.0
    social_rank: int = 0
    face_center: tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))
    landmark_data: dict = field(default_factory=dict)
    name: Optional[str] = None  # Set by face recognition database

    def to_dict(self) -> dict:
        out = {
            "person_id": self.person_id,
            "bbox": self.bbox,
            "estimated_depth": round(self.estimated_depth, 2),
            "body_orientation": self.body_orientation,
            "expression": self.expression,
            "face_area_px": self.face_area_px,
            "social_engagement_score": round(self.social_engagement_score, 1),
            "social_rank": self.social_rank,
        }
        if self.name:
            out["name"] = self.name
        if self.landmark_data:
            out["landmark_data"] = self.landmark_data
        return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REFERENCE_PERSON_AREA = 60_000  # px^2 baseline for person bbox area


def estimate_depth(person_area_px: int) -> float:
    if person_area_px <= 0:
        return 10.0
    depth = math.sqrt(REFERENCE_PERSON_AREA / person_area_px) * 3.0
    return round(min(max(depth, 0.1), 10.0), 2)


def classify_orientation(pose_landmarks: list | None, img_w: int, img_h: int) -> str:
    if not pose_landmarks:
        return "UNKNOWN"
    ls = pose_landmarks.get(5)  # left shoulder
    rs = pose_landmarks.get(6)  # right shoulder
    if not ls or not rs:
        return "UNKNOWN"
    vis_ls = ls[2]
    vis_rs = rs[2]
    if vis_ls < 0.2 or vis_rs < 0.2:
        return "UNKNOWN"

    shoulder_dx = rs[0] - ls[0]
    shoulder_dy = rs[1] - ls[1]
    shoulder_width = math.sqrt(shoulder_dx * shoulder_dx + shoulder_dy * shoulder_dy)
    if shoulder_width < 10:
        return "TURNED_AWAY"

    nose = pose_landmarks.get(0)  # nose
    if nose and nose[2] >= 0.2:
        mid_x = (ls[0] + rs[0]) / 2.0
        offset = nose[0] - mid_x
        if abs(offset) <= shoulder_width * 0.08:
            return "FACING_CAMERA"
        return "PROFILE_RIGHT" if offset > 0 else "PROFILE_LEFT"
    return "ANGLED"


def _resize_for_detection(img: np.ndarray, max_dim: int = 1280) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img, 1.0
    scale = max_dim / max(h, w)
    resized = cv2.resize(img, (int(w * scale), int(h * scale)))
    return resized, scale


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter == 0:
        return 0.0

    union = aw * ah + bw * bh - inter
    return inter / (union + 1e-6)


def _nms_candidates(candidates: list[dict], iou_threshold: float = 0.5, max_count: int = 20) -> list[dict]:
    if not candidates:
        return []
    sorted_candidates = sorted(candidates, key=lambda c: c["score"], reverse=True)
    kept: list[dict] = []
    for c in sorted_candidates:
        if all(_iou(c["bbox"], k["bbox"]) < iou_threshold for k in kept):
            kept.append(c)
        if len(kept) >= max_count:
            break
    return kept


def _is_reasonable_person_bbox(
    bbox: tuple[int, int, int, int],
    img_w: int,
    img_h: int,
) -> bool:
    x, y, w, h = bbox
    if w < 24 or h < 40:
        return False
    if w > img_w * 0.95 or h > img_h * 0.99:
        return False
    area = w * h
    if area < img_w * img_h * 0.0008:
        return False
    aspect = w / max(h, 1)
    if aspect < 0.15 or aspect > 1.9:  # relaxed upper bound for leaning/sitting poses
        return False
    return True


def _norm(vx: float, vy: float) -> tuple[float, float] | None:
    mag = math.sqrt(vx * vx + vy * vy)
    if mag < 1e-6:
        return None
    return (vx / mag, vy / mag)


def _attention_vector_from_keypoints(kps: dict[int, tuple[float, float, float]]) -> tuple[float, float] | None:
    """Estimate in-image attention direction from face/upper-body keypoints."""
    nose = kps.get(0)
    le = kps.get(1)
    re = kps.get(2)
    ls = kps.get(5)
    rs = kps.get(6)

    if nose and le and re and le[2] >= 0.2 and re[2] >= 0.2 and nose[2] >= 0.2:
        eye_mid_x = (le[0] + re[0]) / 2.0
        eye_mid_y = (le[1] + re[1]) / 2.0
        v = _norm(nose[0] - eye_mid_x, nose[1] - eye_mid_y)
        if v:
            return v

    # Fallback: shoulder to nose direction.
    if nose and ls and rs and ls[2] >= 0.2 and rs[2] >= 0.2 and nose[2] >= 0.2:
        sh_mid_x = (ls[0] + rs[0]) / 2.0
        sh_mid_y = (ls[1] + rs[1]) / 2.0
        v = _norm(nose[0] - sh_mid_x, nose[1] - sh_mid_y)
        if v:
            return v
    return None


def _body_focus_vector_from_keypoints(
    kps: dict[int, tuple[float, float, float]],
    orientation: str,
) -> tuple[float, float] | None:
    """
    Body/torso intent vector from pose.
    Keeps pose as primary signal, independent from head gaze.
    """
    ls = kps.get(5)
    rs = kps.get(6)
    lh = kps.get(11)
    rh = kps.get(12)
    nose = kps.get(0)

    # Torso axis (hip -> shoulder) gives body upright direction.
    if ls and rs and lh and rh and ls[2] >= 0.2 and rs[2] >= 0.2 and lh[2] >= 0.2 and rh[2] >= 0.2:
        sh_mid_x = (ls[0] + rs[0]) / 2.0
        sh_mid_y = (ls[1] + rs[1]) / 2.0
        hip_mid_x = (lh[0] + rh[0]) / 2.0
        hip_mid_y = (lh[1] + rh[1]) / 2.0
        torso_v = _norm(sh_mid_x - hip_mid_x, sh_mid_y - hip_mid_y)
        if torso_v:
            if orientation == "PROFILE_RIGHT":
                return _norm(0.8, torso_v[1] * 0.5)
            if orientation == "PROFILE_LEFT":
                return _norm(-0.8, torso_v[1] * 0.5)
            return torso_v

    if nose and ls and rs and nose[2] >= 0.2 and ls[2] >= 0.2 and rs[2] >= 0.2:
        sh_mid_x = (ls[0] + rs[0]) / 2.0
        sh_mid_y = (ls[1] + rs[1]) / 2.0
        v = _norm(nose[0] - sh_mid_x, nose[1] - sh_mid_y)
        if v:
            return v
    return None


def _best_pose_for_bbox(
    bbox: tuple[int, int, int, int],
    pose_dets: list[dict],
    iou_threshold: float = 0.2,
) -> dict | None:
    best = None
    best_iou = 0.0
    for p in pose_dets:
        iou = _iou(bbox, p["bbox"])
        if iou > best_iou:
            best_iou = iou
            best = p
    if best and best_iou >= iou_threshold:
        return best
    return None


def _hog_detect_persons(image_bgr: np.ndarray, hog: cv2.HOGDescriptor) -> list[dict]:
    rects, weights = hog.detectMultiScale(
        image_bgr,
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.05,
    )
    out: list[dict] = []
    for (x, y, w, h), conf in zip(rects, weights):
        out.append({
            "bbox": (int(x), int(y), int(w), int(h)),
            "score": float(conf),
            "source": "hog",
        })
    return out


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

class CVPipeline:
    def __init__(self):
        self._yolo_model_name = os.getenv("YOLO_MODEL", "yolov8n.pt")
        self._yolo_pose_model_name = os.getenv("YOLO_POSE_MODEL", "yolov8n-pose.pt")
        self._yolo_conf = float(os.getenv("YOLO_CONF", "0.25"))
        self._yolo_iou = float(os.getenv("YOLO_IOU", "0.45"))
        self._yolo_pose_conf = float(os.getenv("YOLO_POSE_CONF", "0.25"))
        self._max_persons = int(os.getenv("YOLO_MAX_PERSONS", "20"))

        self._yolo = None
        self._yolo_pose = None
        self._head_gaze = HeadGazeEstimator()
        if YOLO is None:
            logger.warning("Ultralytics not installed. YOLO detector unavailable.")
        else:
            try:
                self._yolo = YOLO(self._yolo_model_name)
                logger.info("YOLO detector ready: %s", self._yolo_model_name)
            except Exception as e:
                logger.warning("YOLO init failed: %s", e)
            try:
                self._yolo_pose = YOLO(self._yolo_pose_model_name)
                logger.info("YOLO pose ready: %s", self._yolo_pose_model_name)
            except Exception as e:
                logger.warning("YOLO pose init failed: %s", e)

        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def _detect_yolo(self, image_bgr: np.ndarray) -> list[dict]:
        if self._yolo is None:
            return []
        try:
            results = self._yolo.predict(
                source=image_bgr,
                classes=[0],  # person
                conf=self._yolo_conf,
                iou=self._yolo_iou,
                verbose=False,
                device="cpu",
            )
        except Exception as e:
            logger.warning("YOLO inference failed: %s", e)
            return []

        detections: list[dict] = []
        if not results:
            return detections

        boxes = results[0].boxes
        if boxes is None:
            return detections

        xyxy = boxes.xyxy.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy().tolist()

        for (x1, y1, x2, y2), conf in zip(xyxy, confs):
            w = max(0, int(x2 - x1))
            h = max(0, int(y2 - y1))
            if w == 0 or h == 0:
                continue
            detections.append(
                {
                    "bbox": (int(x1), int(y1), w, h),
                    "score": float(conf),
                    "source": "yolo",
                }
            )
        return detections

    def _detect_yolo_pose(self, image_bgr: np.ndarray) -> list[dict]:
        if self._yolo_pose is None:
            return []
        try:
            results = self._yolo_pose.predict(
                source=image_bgr,
                classes=[0],  # person
                conf=self._yolo_pose_conf,
                iou=self._yolo_iou,
                verbose=False,
                device="cpu",
            )
        except Exception as e:
            logger.warning("YOLO pose inference failed: %s", e)
            return []

        detections: list[dict] = []
        if not results:
            return detections

        r0 = results[0]
        if r0.boxes is None:
            return detections
        xyxy = r0.boxes.xyxy.cpu().numpy().astype(int)
        confs = r0.boxes.conf.cpu().numpy().tolist()

        kps_xy = None
        kps_conf = None
        if r0.keypoints is not None:
            if r0.keypoints.xy is not None:
                kps_xy = r0.keypoints.xy.cpu().numpy()
            if getattr(r0.keypoints, "conf", None) is not None:
                kps_conf = r0.keypoints.conf.cpu().numpy()

        for idx, ((x1, y1, x2, y2), conf) in enumerate(zip(xyxy, confs)):
            w = max(0, int(x2 - x1))
            h = max(0, int(y2 - y1))
            if w == 0 or h == 0:
                continue
            keypoints: dict[int, tuple[float, float, float]] = {}
            if kps_xy is not None and idx < len(kps_xy):
                for k in range(len(kps_xy[idx])):
                    xk = float(kps_xy[idx][k][0])
                    yk = float(kps_xy[idx][k][1])
                    ck = float(kps_conf[idx][k]) if kps_conf is not None and idx < len(kps_conf) else 1.0
                    keypoints[k] = (xk, yk, ck)

            detections.append(
                {
                    "bbox": (int(x1), int(y1), w, h),
                    "score": float(conf),
                    "source": "yolo_pose",
                    "keypoints": keypoints,
                }
            )
        return detections

    def debug_human_detection(self, image_bgr: np.ndarray) -> dict:
        work_img, det_scale = _resize_for_detection(image_bgr, max_dim=1600)
        work_h, work_w = work_img.shape[:2]

        raw_yolo = self._detect_yolo(work_img)
        raw_pose = self._detect_yolo_pose(work_img)
        raw_hog = _hog_detect_persons(work_img, self._hog)

        accepted: list[dict] = []
        if raw_yolo:
            for d in raw_yolo:
                if d["score"] < self._yolo_conf:
                    continue
                if not _is_reasonable_person_bbox(d["bbox"], work_w, work_h):
                    continue
                pose_match = _best_pose_for_bbox(d["bbox"], raw_pose, iou_threshold=0.18)
                keypoints = pose_match.get("keypoints", {}) if pose_match else {}
                d2 = dict(d)
                d2["keypoints"] = keypoints
                orientation = classify_orientation(keypoints, work_w, work_h)
                d2["body_orientation"] = orientation
                d2["body_focus_vector"] = _body_focus_vector_from_keypoints(keypoints, orientation)

                # Head gaze channel: optional model first, geometric fallback.
                head_gaze = self._head_gaze.estimate(
                    image_bgr=work_img,
                    bbox=d["bbox"],
                    keypoints=keypoints,
                )
                d2["head_gaze_vector"] = head_gaze.get("vector") or _attention_vector_from_keypoints(keypoints)
                d2["head_gaze_source"] = head_gaze.get("source", "keypoint_fallback")
                accepted.append(d2)
        elif raw_pose:
            for d in raw_pose:
                if d["score"] < self._yolo_pose_conf:
                    continue
                if not _is_reasonable_person_bbox(d["bbox"], work_w, work_h):
                    continue
                d2 = {
                    "bbox": d["bbox"],
                    "score": d["score"],
                    "source": "yolo_pose",
                    "keypoints": d.get("keypoints", {}),
                }
                orientation = classify_orientation(d2["keypoints"], work_w, work_h)
                d2["body_orientation"] = orientation
                d2["body_focus_vector"] = _body_focus_vector_from_keypoints(d2["keypoints"], orientation)
                head_gaze = self._head_gaze.estimate(
                    image_bgr=work_img,
                    bbox=d["bbox"],
                    keypoints=d2["keypoints"],
                )
                d2["head_gaze_vector"] = head_gaze.get("vector") or _attention_vector_from_keypoints(d2["keypoints"])
                d2["head_gaze_source"] = head_gaze.get("source", "keypoint_fallback")
                accepted.append(d2)
        else:
            # Fallback only when YOLO has no output.
            for d in raw_hog:
                if d["score"] < 0.5:
                    continue
                if not _is_reasonable_person_bbox(d["bbox"], work_w, work_h):
                    continue
                d2 = dict(d)
                d2["body_orientation"] = "UNKNOWN"
                d2["body_focus_vector"] = None
                d2["head_gaze_vector"] = None
                d2["head_gaze_source"] = "none"
                d2["keypoints"] = {}
                accepted.append(d2)

        after_nms = _nms_candidates(accepted, iou_threshold=0.5, max_count=self._max_persons)
        final_candidates = sorted(after_nms, key=lambda c: c["bbox"][0])

        by_source: dict[str, int] = {}
        for c in accepted:
            s = c.get("source", "unknown")
            by_source[s] = by_source.get(s, 0) + 1

        logger.info(
            "Human candidates: raw_yolo=%d raw_hog=%d accepted=%d by_source=%s final=%d",
            len(raw_yolo),
            len(raw_hog),
            len(accepted),
            by_source,
            len(final_candidates),
        )

        return {
            "work_img": work_img,
            "det_scale": det_scale,
            "work_w": work_w,
            "work_h": work_h,
            "raw_yolo": raw_yolo,
            "raw_pose": raw_pose,
            "raw_hog": raw_hog,
            "accepted": accepted,
            "final_candidates": final_candidates,
            "by_source": by_source,
        }

    def process(self, image_bgr: np.ndarray) -> list[PersonData]:
        img_h, img_w = image_bgr.shape[:2]
        logger.info("Processing image %dx%d", img_w, img_h)

        dbg = self.debug_human_detection(image_bgr)
        det_scale = dbg["det_scale"]
        final_candidates = dbg["final_candidates"]

        if not final_candidates:
            logger.info("No humans detected")
            return []

        persons: list[PersonData] = []
        for idx, det in enumerate(final_candidates, start=1):
            x, y, w, h = det["bbox"]
            orig_x = int(x / det_scale)
            orig_y = int(y / det_scale)
            orig_w = int(w / det_scale)
            orig_h = int(h / det_scale)

            area = max(orig_w * orig_h, 1)
            persons.append(
                PersonData(
                    person_id=f"P{idx}",
                    bbox=[orig_x, orig_y, orig_w, orig_h],
                    estimated_depth=estimate_depth(area),
                    body_orientation=det.get("body_orientation", "UNKNOWN"),
                    expression="UNKNOWN",
                    face_area_px=area,
                    face_center=(orig_x + orig_w / 2, orig_y + orig_h / 2),
                    landmark_data={
                        "source": det.get("source", "unknown"),
                        "conf": round(det.get("score", 0.0), 3),
                        "body_focus_vector": det.get("body_focus_vector"),
                        "head_gaze_vector": det.get("head_gaze_vector"),
                        "head_gaze_source": det.get("head_gaze_source"),
                    },
                )
            )

        logger.info("Returning %d person(s): %s", len(persons), [p.person_id for p in persons])
        return persons

    def close(self):
        # Ultralytics model does not require explicit close.
        return


# ---------------------------------------------------------------------------
# Annotated image rendering
# ---------------------------------------------------------------------------

EXPRESSION_EMOJI = {
    "NEUTRAL": "",
    "SMILING": ":)",
    "TALKING": "...",
    "SURPRISED": "O_O",
    "FOCUSED": ">_<",
    "LAUGHING": ":D",
    "UNKNOWN": "?",
}

SCORE_COLORS = {
    "high": (0, 255, 136),
    "medium": (0, 170, 255),
    "low": (68, 68, 255),
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
        color = (255, 140, 0) if p.name else score_color(p.social_engagement_score)  # blue for known, score-based for unknown

        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

        display_id = p.name if p.name else p.person_id
        label = (
            f"{display_id} "
            f"{EXPRESSION_EMOJI.get(p.expression, '')} "
            f"score:{p.social_engagement_score:.0f}"
        )
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y = max(y, th + 10)
        cv2.rectangle(annotated, (x, label_y - th - 8), (x + tw + 4, label_y), color, -1)
        cv2.putText(
            annotated,
            label,
            (x + 2, label_y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

        depth_label = f"d:{p.estimated_depth:.1f}m"
        cv2.putText(
            annotated,
            depth_label,
            (x + 2, y + h - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )

    return annotated
