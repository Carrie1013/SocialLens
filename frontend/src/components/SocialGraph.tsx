import React, { useEffect, useRef } from "react";
import { AnalysisResult, SCORE_COLOR } from "../types";

interface Props {
  result: AnalysisResult;
  selectedPersonId: string | null;
  onPersonClick: (id: string) => void;
}

const NODE_RADIUS = 24;
const CANVAS_SIZE = 300;

export const SocialGraph: React.FC<Props> = ({
  result,
  selectedPersonId,
  onPersonClick,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const nodePositions = useRef<Record<string, [number, number]>>({});

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !result.persons.length) return;
    const ctx = canvas.getContext("2d")!;
    const size = CANVAS_SIZE;
    canvas.width = size;
    canvas.height = size;
    ctx.clearRect(0, 0, size, size);

    const persons = result.persons;
    const n = persons.length;
    const cx = size / 2;
    const cy = size / 2;
    const radius = Math.min(size / 2 - NODE_RADIUS - 16, 110);

    // Arrange nodes in a circle
    const positions: Record<string, [number, number]> = {};
    persons.forEach((p, i) => {
      const angle = (2 * Math.PI * i) / n - Math.PI / 2;
      positions[p.person_id] = [
        cx + radius * Math.cos(angle),
        cy + radius * Math.sin(angle),
      ];
    });
    nodePositions.current = positions;

    // Draw group connection lines (guard against missing data)
    const groups = result.groups || [];
    groups.forEach((grp, gi) => {
      const groupColors = ["#00ff88", "#00aaff", "#ffaa00", "#ff4444"];
      const lineColor = groupColors[gi % groupColors.length];
      const members = grp.member_ids;
      for (let a = 0; a < members.length; a++) {
        for (let b = a + 1; b < members.length; b++) {
          const pa = positions[members[a]];
          const pb = positions[members[b]];
          if (!pa || !pb) continue;
          ctx.save();
          ctx.strokeStyle = lineColor;
          ctx.lineWidth = 1.5;
          ctx.globalAlpha = 0.5;
          ctx.setLineDash([4, 3]);
          ctx.beginPath();
          ctx.moveTo(pa[0], pa[1]);
          ctx.lineTo(pb[0], pb[1]);
          ctx.stroke();
          ctx.restore();
        }
      }
    });

    // Draw nodes
    persons.forEach((p) => {
      const [nx, ny] = positions[p.person_id];
      const color = SCORE_COLOR(p.social_engagement_score);
      const isSelected = p.person_id === selectedPersonId;
      const isCenter = result.social_center === p.person_id;

      // Glow for selected
      if (isSelected) {
        ctx.save();
        ctx.shadowColor = "#ffffff";
        ctx.shadowBlur = 16;
        ctx.beginPath();
        ctx.arc(nx, ny, NODE_RADIUS + 3, 0, 2 * Math.PI);
        ctx.fillStyle = "rgba(255,255,255,0.15)";
        ctx.fill();
        ctx.restore();
      }

      // Outer ring
      ctx.save();
      ctx.beginPath();
      ctx.arc(nx, ny, NODE_RADIUS, 0, 2 * Math.PI);
      ctx.fillStyle = "#0f0f1a";
      ctx.fill();
      ctx.strokeStyle = isCenter ? "#ffd700" : color;
      ctx.lineWidth = isSelected ? 3 : 2;
      ctx.stroke();
      ctx.restore();

      // Score arc
      ctx.save();
      ctx.beginPath();
      ctx.arc(
        nx, ny, NODE_RADIUS - 4,
        -Math.PI / 2,
        -Math.PI / 2 + (2 * Math.PI * p.social_engagement_score) / 100,
      );
      ctx.strokeStyle = color;
      ctx.lineWidth = 4;
      ctx.lineCap = "round";
      ctx.stroke();
      ctx.restore();

      // Label
      ctx.save();
      ctx.font = "bold 11px monospace";
      ctx.fillStyle = color;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(p.person_id, nx, ny);
      ctx.restore();

      // Score below
      ctx.save();
      ctx.font = "10px monospace";
      ctx.fillStyle = "#888";
      ctx.textAlign = "center";
      ctx.fillText(`${p.social_engagement_score}`, nx, ny + NODE_RADIUS + 10);
      ctx.restore();

      // Crown for center
      if (isCenter) {
        ctx.save();
        ctx.font = "14px sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("👑", nx, ny - NODE_RADIUS - 8);
        ctx.restore();
      }
    });
  }, [result, selectedPersonId]);

  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mx = (e.clientX - rect.left) * (CANVAS_SIZE / rect.width);
    const my = (e.clientY - rect.top) * (CANVAS_SIZE / rect.height);
    for (const [pid, [nx, ny]] of Object.entries(nodePositions.current)) {
      const dx = mx - nx;
      const dy = my - ny;
      if (Math.sqrt(dx * dx + dy * dy) <= NODE_RADIUS) {
        onPersonClick(pid);
        return;
      }
    }
  };

  if (!result.persons.length) return null;

  return (
    <div>
      <div className="text-xs font-mono text-gray-500 uppercase mb-2">Social Graph</div>
      <div className="flex justify-center">
        <canvas
          ref={canvasRef}
          width={CANVAS_SIZE}
          height={CANVAS_SIZE}
          className="cursor-pointer rounded-lg bg-gray-900/50 border border-gray-800"
          style={{ width: CANVAS_SIZE, height: CANVAS_SIZE }}
          onClick={handleClick}
        />
      </div>
    </div>
  );
};
