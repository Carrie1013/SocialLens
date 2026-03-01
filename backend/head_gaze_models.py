"""
Optional head/gaze model adapter.

Priority:
1) 6DRepNet head pose (if available)
2) Gaze-LLE / Sharingan style estimators (placeholder hooks)
3) Fallback to None (caller will use keypoint geometry)

This file is intentionally defensive: if dependencies are missing, pipeline
continues with geometric fallback.
"""

from __future__ import annotations

import importlib
import logging
import math
import os
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _norm(vx: float, vy: float) -> tuple[float, float] | None:
    mag = math.sqrt(vx * vx + vy * vy)
    if mag < 1e-6:
        return None
    return (vx / mag, vy / mag)


class HeadGazeEstimator:
    def __init__(self):
        self._backend = "none"
        self._model: Any = None
        self._enabled = os.getenv("ENABLE_HEAD_GAZE_MODEL", "1") == "1"
        if not self._enabled:
            logger.info("HeadGazeEstimator disabled by ENABLE_HEAD_GAZE_MODEL=0")
            return

        # Try 6DRepNet first.
        try:
            sixd = importlib.import_module("sixdrepnet")
            cls = getattr(sixd, "SixDRepNet", None)
            if cls is not None:
                self._model = cls()
                self._backend = "6drepnet"
                logger.info("HeadGazeEstimator backend: 6drepnet")
                return
        except Exception as e:
            logger.debug("6drepnet unavailable: %s", e)

        # Placeholder hooks for gaze-following packages. API differs across repos.
        try:
            importlib.import_module("gazelle")
            self._backend = "gazelle_stub"
            logger.info("HeadGazeEstimator backend: gazelle_stub (no direct wrapper yet)")
            return
        except Exception:
            pass

        try:
            importlib.import_module("sharingan")
            self._backend = "sharingan_stub"
            logger.info("HeadGazeEstimator backend: sharingan_stub (no direct wrapper yet)")
            return
        except Exception:
            pass

        logger.info("HeadGazeEstimator backend: none (geometric fallback)")

    def backend(self) -> str:
        return self._backend

    def estimate(
        self,
        *,
        image_bgr: np.ndarray,
        bbox: tuple[int, int, int, int],
        keypoints: dict[int, tuple[float, float, float]],
    ) -> dict:
        """
        Returns:
          {"vector": (vx, vy) | None, "source": str, "yaw": float|None, "pitch": float|None}
        """
        if self._backend == "6drepnet" and self._model is not None:
            out = self._estimate_6drepnet(image_bgr=image_bgr, bbox=bbox)
            if out["vector"] is not None:
                return out

        # No strong model output available.
        return {"vector": None, "source": "none", "yaw": None, "pitch": None}

    def _estimate_6drepnet(self, *, image_bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> dict:
        """
        Best-effort 6DRepNet wrapper.
        Different forks expose different APIs, so we keep this loose and safe.
        """
        x, y, w, h = bbox
        h_img, w_img = image_bgr.shape[:2]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w_img, x + w)
        y2 = min(h_img, y + h)
        if x2 <= x1 or y2 <= y1:
            return {"vector": None, "source": "6drepnet", "yaw": None, "pitch": None}
        crop = image_bgr[y1:y2, x1:x2]

        yaw = None
        pitch = None
        try:
            # Common pattern: model.predict returns (pitch, yaw, roll)
            if hasattr(self._model, "predict"):
                pred = self._model.predict(crop)
                if isinstance(pred, (list, tuple)) and len(pred) >= 2:
                    pitch = float(pred[0])
                    yaw = float(pred[1])
            # Some wrappers expose call operator.
            elif callable(self._model):
                pred = self._model(crop)
                if isinstance(pred, (list, tuple)) and len(pred) >= 2:
                    pitch = float(pred[0])
                    yaw = float(pred[1])
        except Exception as e:
            logger.debug("6drepnet inference failed: %s", e)
            return {"vector": None, "source": "6drepnet", "yaw": None, "pitch": None}

        if yaw is None or pitch is None:
            return {"vector": None, "source": "6drepnet", "yaw": None, "pitch": None}

        # Map yaw/pitch to image-plane vector (heuristic projection).
        vx = math.sin(math.radians(yaw))
        vy = -math.sin(math.radians(pitch))
        v = _norm(vx, vy)
        return {"vector": v, "source": "6drepnet", "yaw": yaw, "pitch": pitch}
