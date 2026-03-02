import React from "react";
import { DominanceResult, PersonData } from "../types";

interface Props {
  result: DominanceResult;
  personA: PersonData;
  personB: PersonData;
  onClose: () => void;
}

function ScoreBar({ score, color }: { score: number; color: string }) {
  return (
    <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
      <div
        className="h-full rounded-full transition-all duration-500"
        style={{ width: `${Math.round(score * 100)}%`, backgroundColor: color }}
      />
    </div>
  );
}

export const DominancePanel: React.FC<Props> = ({ result, personA, personB, onClose }) => {
  const labelA = personA.name ?? personA.person_id;
  const labelB = personB.name ?? personB.person_id;

  const dominantLabel =
    result.dominant_person === "A" ? labelA :
    result.dominant_person === "B" ? labelB :
    "Equal";

  const dominantColor =
    result.dominant_person === "A" ? "#ff4444" :
    result.dominant_person === "B" ? "#00ffff" :
    "#ffaa00";

  const engagementColor =
    result.engagement_score >= 0.6 ? "#00ff88" :
    result.engagement_score >= 0.35 ? "#ffaa00" :
    "#ff4444";

  return (
    <div className="glass-panel rounded-xl p-4 border border-purple-800/60 bg-purple-950/20">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <span className="text-xs font-mono text-purple-400 uppercase tracking-wider">
          ⚡ Dominance Analysis
        </span>
        <button
          onClick={onClose}
          className="text-gray-600 hover:text-gray-300 text-xs transition-colors"
        >
          ✕
        </button>
      </div>

      {/* Dominant person callout */}
      <div className="mb-3 rounded-lg px-3 py-2 bg-gray-900/60 border border-gray-700 text-center">
        <div className="text-xs text-gray-500 font-mono mb-0.5">Dominant</div>
        <div className="font-mono font-bold text-sm" style={{ color: dominantColor }}>
          {result.dominant_person === "equal" ? "⚖ Equal" : `⚡ ${dominantLabel}`}
        </div>
      </div>

      {/* A vs B dominance scores */}
      <div className="grid grid-cols-2 gap-2 mb-3">
        {[
          { label: labelA, role: "A", score: result.dominance_score_a, color: "#ff4444", bl: result.body_language_a },
          { label: labelB, role: "B", score: result.dominance_score_b, color: "#00ffff", bl: result.body_language_b },
        ].map(({ label, role, score, color, bl }) => (
          <div key={role} className="rounded-lg p-2 bg-gray-900/40 border border-gray-800">
            <div className="flex items-center gap-1.5 mb-1">
              <span
                className="text-xs font-mono font-bold px-1.5 py-0.5 rounded"
                style={{ color, borderColor: color, border: `1px solid ${color}` }}
              >
                {role}
              </span>
              <span className="text-xs font-mono truncate" style={{ color }}>{label}</span>
            </div>
            <div className="flex justify-between text-xs font-mono mb-1">
              <span className="text-gray-500">dominance</span>
              <span style={{ color }}>{Math.round(score * 100)}%</span>
            </div>
            <ScoreBar score={score} color={color} />
            <div className="mt-1.5 text-[10px] text-gray-500 font-mono leading-tight">{bl}</div>
          </div>
        ))}
      </div>

      {/* Engagement score */}
      <div className="mb-3">
        <div className="flex justify-between text-xs font-mono mb-1">
          <span className="text-gray-400">Mutual Engagement</span>
          <span style={{ color: engagementColor }}>{Math.round(result.engagement_score * 100)}%</span>
        </div>
        <ScoreBar score={result.engagement_score} color={engagementColor} />
      </div>

      {/* Interaction type + dynamic */}
      <div className="mb-2 flex flex-col gap-0.5">
        {result.interaction_type && (
          <div className="flex items-center gap-1.5">
            <span className="text-[10px] font-mono text-gray-600 uppercase">Interaction</span>
            <span className="text-[10px] font-mono text-gray-400 bg-gray-800 px-1.5 py-0.5 rounded">
              {result.interaction_type}
            </span>
          </div>
        )}
        <div>
          <span className="text-[10px] font-mono text-gray-600 uppercase">Dynamic</span>
          <div className="text-xs font-mono text-purple-300 mt-0.5 italic">
            "{result.relationship_dynamic}"
          </div>
        </div>
      </div>

      {/* Reasoning */}
      <div className="pt-2 border-t border-gray-800 text-xs text-gray-400 leading-relaxed">
        {result.reasoning}
      </div>
    </div>
  );
};
