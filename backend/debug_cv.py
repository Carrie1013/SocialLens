"""
Offline CV debugger (CV only, no external APIs).

Usage:
  ./.venv/bin/python debug_cv.py --image /path/to/image.jpg
  ./.venv/bin/python debug_cv.py --image /path/to/image.jpg --out-dir ./debug_out
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2

from cv_pipeline import CVPipeline, PersonData, classify_orientation, draw_annotations, estimate_depth


SOURCE_COLOR = {
    "yolo": (0, 255, 136),
    "yolo_pose": (255, 80, 180),
    "hog": (255, 170, 0),
}


def _draw_boxes(image_bgr, detections: list[dict], title: str):
    out = image_bgr.copy()
    for i, det in enumerate(detections, start=1):
        x, y, w, h = det["bbox"]
        src = det.get("source", "?")
        score = det.get("score", 0.0)
        color = SOURCE_COLOR.get(src, (255, 255, 255))
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        label = f"{i}:{src}:{score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ly = max(y, th + 8)
        cv2.rectangle(out, (x, ly - th - 6), (x + tw + 3, ly), color, -1)
        cv2.putText(out, label, (x + 1, ly - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    cv2.putText(out, title, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return out


def _draw_pose_and_attention(image_bgr, detections: list[dict], title: str):
    out = image_bgr.copy()
    for i, det in enumerate(detections, start=1):
        x, y, w, h = det["bbox"]
        src = det.get("source", "?")
        score = det.get("score", 0.0)
        color = SOURCE_COLOR.get(src, (255, 255, 255))
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)

        label = f"{i}:{src}:{score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ly = max(y, th + 8)
        cv2.rectangle(out, (x, ly - th - 6), (x + tw + 3, ly), color, -1)
        cv2.putText(out, label, (x + 1, ly - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

        # Draw pose keypoints if available.
        kps = det.get("keypoints") or {}
        if isinstance(kps, dict):
            for _, v in kps.items():
                if not isinstance(v, (list, tuple)) or len(v) < 3:
                    continue
                kx, ky, kc = float(v[0]), float(v[1]), float(v[2])
                if kc < 0.2:
                    continue
                cv2.circle(out, (int(kx), int(ky)), 2, (80, 255, 255), -1)

        # Draw body focus (green) and head gaze (yellow) arrows from bbox center.
        body_v = det.get("body_focus_vector")
        if isinstance(body_v, (list, tuple)) and len(body_v) == 2:
            vx, vy = float(body_v[0]), float(body_v[1])
            mag = math.sqrt(vx * vx + vy * vy)
            if mag > 1e-6:
                vx /= mag
                vy /= mag
                cx = x + w // 2
                cy = y + h // 2
                arrow_len = int(max(28, min(80, 0.35 * min(w, h))))
                ex = int(cx + vx * arrow_len)
                ey = int(cy + vy * arrow_len)
                cv2.arrowedLine(
                    out,
                    (cx, cy),
                    (ex, ey),
                    (0, 255, 136),
                    2,
                    tipLength=0.22,
                )

        head_v = det.get("head_gaze_vector")
        if isinstance(head_v, (list, tuple)) and len(head_v) == 2:
            vx, vy = float(head_v[0]), float(head_v[1])
            mag = math.sqrt(vx * vx + vy * vy)
            if mag > 1e-6:
                vx /= mag
                vy /= mag
                cx = x + w // 2
                cy = y + h // 2
                arrow_len = int(max(28, min(80, 0.45 * min(w, h))))
                ex = int(cx + vx * arrow_len)
                ey = int(cy + vy * arrow_len)
                cv2.arrowedLine(
                    out,
                    (cx, cy),
                    (ex, ey),
                    (0, 255, 255),
                    2,
                    tipLength=0.25,
                )

    cv2.putText(out, title, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return out


def _to_serializable(dets: list[dict]) -> list[dict]:
    return [
        {
            "bbox": [int(v) for v in d["bbox"]],
            "score": round(float(d.get("score", 0.0)), 4),
            "source": d.get("source"),
        }
        for d in dets
    ]


def _persons_from_final(final_dets: list[dict], det_scale: float, work_w: int, work_h: int) -> list[PersonData]:
    persons: list[PersonData] = []
    for idx, det in enumerate(final_dets, start=1):
        x, y, w, h = det["bbox"]
        pose_lms = det.get("pose_landmarks")
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
                body_orientation=classify_orientation(pose_lms, work_w, work_h) if pose_lms else "UNKNOWN",
                expression="UNKNOWN",
                face_area_px=area,
                face_center=(orig_x + orig_w / 2, orig_y + orig_h / 2),
                landmark_data={"source": det.get("source")},
            )
        )
    return persons


def main():
    parser = argparse.ArgumentParser(description="Run CV-only debug pipeline on one image.")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--out-dir", default="debug_out", help="Output directory")
    args = parser.parse_args()

    image_path = Path(args.image).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise SystemExit(f"Failed to read image: {image_path}")

    pipeline = CVPipeline()
    try:
        dbg = pipeline.debug_human_detection(image_bgr)
    finally:
        pipeline.close()

    work_img = dbg["work_img"]
    det_scale = dbg["det_scale"]
    work_w = dbg["work_w"]
    work_h = dbg["work_h"]
    raw_yolo = dbg["raw_yolo"]
    raw_pose = dbg.get("raw_pose", [])
    raw_hog = dbg["raw_hog"]
    accepted = dbg["accepted"]
    final_dets = dbg["final_candidates"]
    source_counts = dbg["by_source"]

    persons = _persons_from_final(final_dets, det_scale, work_w, work_h)
    annotated = draw_annotations(image_bgr, persons)

    base = image_path.stem
    json_path = out_dir / f"{base}.cv_debug.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "image": str(image_path),
                "counts": {
                    "raw_yolo": len(raw_yolo),
                    "raw_pose": len(raw_pose),
                    "raw_hog": len(raw_hog),
                    "accepted": len(accepted),
                    "final": len(final_dets),
                },
                "raw_source_counts": source_counts,
                "final_persons": [
                    {
                        "person_id": p.person_id,
                        "bbox": p.bbox,
                        "source": p.landmark_data.get("source"),
                        "depth": p.estimated_depth,
                        "orientation": p.body_orientation,
                    }
                    for p in persons
                ],
                "raw_yolo": _to_serializable(raw_yolo),
                "raw_pose": _to_serializable(raw_pose),
                "raw_hog": _to_serializable(raw_hog),
                "accepted": _to_serializable(accepted),
                "final_candidates": _to_serializable(final_dets),
                "accepted_with_pose_attention": accepted,
                "final_with_pose_attention": final_dets,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    cv2.imwrite(str(out_dir / f"{base}.raw_yolo.jpg"), _draw_boxes(work_img, raw_yolo, "RAW YOLO"))
    cv2.imwrite(str(out_dir / f"{base}.raw_pose.jpg"), _draw_boxes(work_img, raw_pose, "RAW YOLO-POSE"))
    cv2.imwrite(str(out_dir / f"{base}.raw_hog.jpg"), _draw_boxes(work_img, raw_hog, "RAW HOG"))
    cv2.imwrite(str(out_dir / f"{base}.accepted.jpg"), _draw_boxes(work_img, accepted, "ACCEPTED"))
    cv2.imwrite(
        str(out_dir / f"{base}.accepted_pose_attention.jpg"),
        _draw_pose_and_attention(work_img, accepted, "ACCEPTED + POSE + ATTENTION"),
    )
    cv2.imwrite(
        str(out_dir / f"{base}.final_pose_attention.jpg"),
        _draw_pose_and_attention(work_img, final_dets, "FINAL + POSE + ATTENTION"),
    )
    cv2.imwrite(str(out_dir / f"{base}.final.jpg"), annotated)

    print(f"Input: {image_path}")
    print(f"Output dir: {out_dir}")
    print(f"raw_yolo: {len(raw_yolo)}")
    print(f"raw_pose: {len(raw_pose)}")
    print(f"raw_hog: {len(raw_hog)}")
    print(f"accepted: {len(accepted)}")
    print(f"final: {len(final_dets)}")
    print(f"source_counts: {source_counts}")
    print(f"Debug json: {json_path}")


if __name__ == "__main__":
    main()
