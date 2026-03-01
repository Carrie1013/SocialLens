"""
Personal face recognition database.

Stores face encodings for known people and identifies them in video frames.
Gracefully degrades if the face_recognition library is not installed.
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
    logger.info("face_recognition loaded OK")
except Exception as e:
    face_recognition = None  # type: ignore
    FACE_RECOGNITION_AVAILABLE = False
    logger.warning("face_recognition unavailable (%s: %s). Personal face ID disabled.",
                   type(e).__name__, e)

DB_DIR = Path(os.getenv("FACE_DB_DIR", "face_db"))
DB_INDEX = DB_DIR / "index.pkl"


class FaceDatabase:
    """Persistent database of named face encodings for person identification."""

    def __init__(self) -> None:
        # dict[name -> list of 128-d face encodings]
        self._encodings: dict[str, list] = {}
        self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not FACE_RECOGNITION_AVAILABLE:
            return
        DB_DIR.mkdir(parents=True, exist_ok=True)
        if DB_INDEX.exists():
            try:
                with open(DB_INDEX, "rb") as f:
                    self._encodings = pickle.load(f)
                logger.info("Loaded face DB with %d person(s): %s",
                            len(self._encodings), list(self._encodings.keys()))
            except Exception as e:
                logger.warning("Failed to load face DB: %s — starting fresh", e)
                self._encodings = {}

    def _save(self) -> None:
        DB_DIR.mkdir(parents=True, exist_ok=True)
        with open(DB_INDEX, "wb") as f:
            pickle.dump(self._encodings, f)

    # ── Public API ────────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Return True if face recognition library is installed."""
        return FACE_RECOGNITION_AVAILABLE

    def add_person(self, name: str, image_bgr: np.ndarray) -> str:
        """
        Register a person by extracting their face encoding from the image.

        Returns:
          "ok"            — face found and stored successfully
          "no_face"       — no face detected in the image
          "not_installed" — face_recognition library not installed
        """
        if not FACE_RECOGNITION_AVAILABLE:
            return "not_installed"

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb)
        if not encodings:
            logger.warning("No face found in registration image for '%s'", name)
            return "no_face"

        if name not in self._encodings:
            self._encodings[name] = []
        self._encodings[name].append(encodings[0])
        self._save()
        logger.info("Registered face for '%s' (%d encoding(s) total)",
                    name, len(self._encodings[name]))
        return "ok"

    def remove_person(self, name: str) -> bool:
        """Remove a person from the database. Returns True if found."""
        if name in self._encodings:
            del self._encodings[name]
            self._save()
            logger.info("Removed '%s' from face DB", name)
            return True
        return False

    def list_persons(self) -> list[str]:
        """Return names of all registered people."""
        return list(self._encodings.keys())

    def identify(
        self,
        image_bgr: np.ndarray,
        bbox: tuple[int, int, int, int],
        threshold: float = 0.55,
    ) -> Optional[str]:
        """
        Try to identify a person from their bounding box region.

        Returns the matched name or None if no confident match is found.
        """
        if not FACE_RECOGNITION_AVAILABLE or not self._encodings:
            return None

        x, y, w, h = bbox
        # Slightly expand the crop so we capture the full face.
        pad = int(min(w, h) * 0.1)
        img_h, img_w = image_bgr.shape[:2]
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img_w, x + w + pad)
        y2 = min(img_h, y + h + pad)
        crop = image_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        unknown_encodings = face_recognition.face_encodings(rgb_crop)
        if not unknown_encodings:
            return None

        unknown_enc = unknown_encodings[0]
        best_name: Optional[str] = None
        best_dist = threshold

        for name, known_encs in self._encodings.items():
            if not known_encs:
                continue
            distances = face_recognition.face_distance(known_encs, unknown_enc)
            min_dist = float(np.min(distances))
            if min_dist < best_dist:
                best_dist = min_dist
                best_name = name

        return best_name
