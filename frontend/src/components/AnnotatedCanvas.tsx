import React, { useEffect, useRef, useState } from "react";
import {
  AnalysisResult,
  PersonData,
  EXPRESSION_EMOJI,
  ROLE_ICON,
  SCORE_COLOR,
} from "../types";

interface Props {
  result: AnalysisResult | null;
  onPersonClick: (personId: string) => void;
  selectedPersonId: string | null;
  isGeneratingVoice: boolean;
}

// Group outline colours (cycling)
const GROUP_COLORS = [
  "rgba(0,255,136,0.15)",
  "rgba(0,170,255,0.15)",
  "rgba(255,170,0,0.15)",
  "rgba(255,68,68,0.15)",
];
const GROUP_BORDER_COLORS = [
  "#00ff88",
  "#00aaff",
  "#ffaa00",
  "#ff4444",
];

const CANVAS_MAX_W = 960;
const CANVAS_MAX_H = 700; // fallback when container height is unknown

export const AnnotatedCanvas: React.FC<Props> = ({
  result,
  onPersonClick,
  selectedPersonId,
  isGeneratingVoice,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const [imgLoaded, setImgLoaded] = useState(false);
  const [scale, setScale] = useState(1);

  // ── Load annotated image ──────────────────────────────────────────────────

  useEffect(() => {
    if (!result?.annotated_image) {
      setImgLoaded(false);
      return;
    }
    const img = new Image();
    img.onload = () => {
      imgRef.current = img;
      setImgLoaded(true);
    };
    img.src = `data:image/jpeg;base64,${result.annotated_image}`;
  }, [result?.annotated_image]);

  // ── Draw canvas ───────────────────────────────────────────────────────────

  useEffect(() => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    const container = containerRef.current;
    if (!canvas || !img || !imgLoaded || !result || !container) return;

    const containerW = container.clientWidth || CANVAS_MAX_W;
    const containerH = container.clientHeight || CANVAS_MAX_H;
    const scaleByW = Math.min(containerW, CANVAS_MAX_W) / img.naturalWidth;
    const scaleByH = (containerH > 50 ? containerH : CANVAS_MAX_H) / img.naturalHeight;
    const scaleX = Math.min(scaleByW, scaleByH);
    const drawW = img.naturalWidth * scaleX;
    const drawH = img.naturalHeight * scaleX;
    setScale(scaleX);

    canvas.width = drawW;
    canvas.height = drawH;

    const ctx = canvas.getContext("2d")!;
    ctx.clearRect(0, 0, drawW, drawH);

    // Background image (already annotated with CV boxes from backend)
    ctx.drawImage(img, 0, 0, drawW, drawH);

    const s = scaleX;

    // ── Group outlines ──────────────────────────────────────────────────────
    result.groups.forEach((grp, gi) => {
      const r = grp.bounding_region;
      const gx = r.x * s;
      const gy = r.y * s;
      const gw = r.w * s;
      const gh = r.h * s;

      ctx.save();
      ctx.strokeStyle = GROUP_BORDER_COLORS[gi % GROUP_BORDER_COLORS.length];
      ctx.fillStyle = GROUP_COLORS[gi % GROUP_COLORS.length];
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 3]);
      ctx.beginPath();
      ctx.roundRect(gx, gy, gw, gh, 8);
      ctx.fill();
      ctx.stroke();
      ctx.restore();

      // Group label
      ctx.save();
      ctx.font = "bold 11px monospace";
      ctx.fillStyle = GROUP_BORDER_COLORS[gi % GROUP_BORDER_COLORS.length];
      ctx.fillText(`${grp.group_id}: ${grp.group_type} (${grp.cohesion_score})`, gx + 4, gy - 4);
      ctx.restore();
    });

    // ── Connection lines between group members ──────────────────────────────
    const personCenters: Record<string, [number, number]> = {};
    result.persons.forEach((p) => {
      const [px, py, pw, ph] = p.bbox;
      personCenters[p.person_id] = [(px + pw / 2) * s, (py + ph / 2) * s];
    });

    result.groups.forEach((grp, gi) => {
      const ids = grp.member_ids;
      for (let a = 0; a < ids.length; a++) {
        for (let b = a + 1; b < ids.length; b++) {
          const ca = personCenters[ids[a]];
          const cb = personCenters[ids[b]];
          if (!ca || !cb) continue;
          ctx.save();
          ctx.strokeStyle = GROUP_BORDER_COLORS[gi % GROUP_BORDER_COLORS.length];
          ctx.lineWidth = 1.5;
          ctx.setLineDash([4, 4]);
          ctx.globalAlpha = 0.6;
          ctx.beginPath();
          ctx.moveTo(ca[0], ca[1]);
          ctx.lineTo(cb[0], cb[1]);
          ctx.stroke();
          ctx.restore();
        }
      }
    });

    // ── Role badges & selected highlight ──────────────────────────────────
    result.persons.forEach((p) => {
      const [px, py, pw, ph] = p.bbox;
      const bx = px * s;
      const by = py * s;
      const bw = pw * s;
      const bh = ph * s;
      const cx = bx + bw / 2;
      const cy = by + bh / 2;

      // Selected highlight
      if (p.person_id === selectedPersonId) {
        ctx.save();
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 3;
        ctx.shadowColor = "#ffffff";
        ctx.shadowBlur = 12;
        ctx.strokeRect(bx - 2, by - 2, bw + 4, bh + 4);
        ctx.restore();
      }

      // Role badge
      const role = result.roles?.[p.person_id];
      if (role) {
        const icon = ROLE_ICON[role.role] ?? "👤";
        ctx.save();
        ctx.font = "14px sans-serif";
        ctx.fillText(icon, bx + bw - 18, by + 16);
        ctx.restore();
      }

      // Expression emoji overlay (bottom-right of box)
      const emoji = EXPRESSION_EMOJI[p.expression] ?? "❓";
      ctx.save();
      ctx.font = "14px sans-serif";
      ctx.fillText(emoji, bx + bw - 18, by + bh - 4);
      ctx.restore();

      // Social center crown
      if (result.social_center === p.person_id) {
        ctx.save();
        ctx.font = "18px sans-serif";
        ctx.fillText("👑", bx + bw / 2 - 9, by - 6);
        ctx.restore();
      }

      // Dual-focus visualization:
      // green = body/pose focus, yellow = head/gaze focus.
      const bodyVec = p.landmark_data?.body_focus_vector;
      if (bodyVec && bodyVec.length === 2) {
        const [vx, vy] = bodyVec;
        const mag = Math.hypot(vx, vy) || 1;
        const len = Math.max(18, Math.min(48, Math.min(bw, bh) * 0.28));
        ctx.save();
        ctx.strokeStyle = "#00ff88";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(cx + (vx / mag) * len, cy + (vy / mag) * len);
        ctx.stroke();
        ctx.restore();
      }

      const headVec = p.landmark_data?.head_gaze_vector;
      if (headVec && headVec.length === 2) {
        const [vx, vy] = headVec;
        const mag = Math.hypot(vx, vy) || 1;
        const len = Math.max(22, Math.min(64, Math.min(bw, bh) * 0.38));
        ctx.save();
        ctx.strokeStyle = "#ffd400";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(cx + (vx / mag) * len, cy + (vy / mag) * len);
        ctx.stroke();
        ctx.restore();
      }
    });

    // On-canvas legend for dual-focus arrows.
    ctx.save();
    const lx = 12;
    const ly = drawH - 30;
    ctx.fillStyle = "rgba(0,0,0,0.55)";
    ctx.fillRect(lx - 8, ly - 18, 240, 24);
    ctx.strokeStyle = "#00ff88";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(lx, ly - 6);
    ctx.lineTo(lx + 20, ly - 6);
    ctx.stroke();
    ctx.fillStyle = "#c8d1dc";
    ctx.font = "12px monospace";
    ctx.fillText("body focus", lx + 26, ly - 2);
    ctx.strokeStyle = "#ffd400";
    ctx.beginPath();
    ctx.moveTo(lx + 112, ly - 6);
    ctx.lineTo(lx + 132, ly - 6);
    ctx.stroke();
    ctx.fillText("head/gaze focus", lx + 138, ly - 2);
    ctx.restore();
  }, [result, imgLoaded, selectedPersonId]);

  // ── Click detection ───────────────────────────────────────────────────────

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!result || !canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;

    for (const p of result.persons) {
      const [px, py, pw, ph] = p.bbox;
      const bx = px * scale;
      const by = py * scale;
      const bw = pw * scale;
      const bh = ph * scale;
      if (cx >= bx && cx <= bx + bw && cy >= by && cy <= by + bh) {
        onPersonClick(p.person_id);
        return;
      }
    }
  };

  return (
    <div
      ref={containerRef}
      className="relative bg-black rounded-lg overflow-hidden border border-gray-700 h-full min-h-[200px] flex items-center justify-center"
    >
      {!result && (
        <div className="text-center text-gray-500 font-mono py-16">
          <div className="text-5xl mb-3">🔍</div>
          <div>Upload or capture an image to begin analysis</div>
        </div>
      )}

      {result && !imgLoaded && (
        <div className="text-gray-400 font-mono animate-pulse">Loading…</div>
      )}

      <canvas
        ref={canvasRef}
        className={`block cursor-crosshair ${imgLoaded ? "" : "hidden"}`}
        onClick={handleCanvasClick}
        style={{ maxWidth: "100%" }}
      />

      {isGeneratingVoice && (
        <div className="absolute inset-0 bg-black/60 flex items-center justify-center">
          <div className="text-green-400 font-mono text-center">
            <div className="text-3xl mb-2 animate-pulse">🎙</div>
            <div>Generating voice…</div>
          </div>
        </div>
      )}
    </div>
  );
};
