import React from "react";
import {
  PersonData,
  AnalysisResult,
  EXPRESSION_EMOJI,
  ROLE_ICON,
  SCORE_COLOR,
} from "../types";

interface Props {
  person: PersonData;
  result: AnalysisResult;
  isSelected: boolean;
  isGeneratingVoice: boolean;
  voiceEnabled: boolean;
  onClick: () => void;
  onVoice: () => void;
}

export const PersonCard: React.FC<Props> = ({
  person,
  result,
  isSelected,
  isGeneratingVoice,
  voiceEnabled,
  onClick,
  onVoice,
}) => {
  const role = result.roles?.[person.person_id];
  const isCenter = result.social_center === person.person_id;
  const color = SCORE_COLOR(person.social_engagement_score);

  return (
    <div
      onClick={onClick}
      className={`
        relative rounded-lg p-3 border cursor-pointer transition-all duration-200
        ${isSelected
          ? "border-white bg-white/10 shadow-lg shadow-white/10"
          : "border-gray-700 bg-gray-900/60 hover:border-gray-500"}
      `}
      style={isSelected ? { borderColor: color } : {}}
    >
      {/* Header row */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <div>
            {person.name && (
              <div className="font-mono font-bold text-sm" style={{ color }}>
                {person.name}
              </div>
            )}
            <span className={`font-mono text-xs ${person.name ? "text-gray-500" : "font-bold text-sm"}`}
                  style={person.name ? {} : { color }}>
              {person.person_id}
            </span>
          </div>
          {isCenter && <span title="Social center">👑</span>}
          {role && (
            <span className="text-xs bg-gray-800 rounded px-1 py-0.5">
              {ROLE_ICON[role.role]} {role.role}
            </span>
          )}
        </div>
        <span className="text-lg" title={person.expression}>
          {EXPRESSION_EMOJI[person.expression]}
        </span>
      </div>

      {/* Score bar */}
      <div className="mb-2">
        <div className="flex justify-between text-xs font-mono mb-1">
          <span className="text-gray-400">Engagement</span>
          <span style={{ color }}>{person.social_engagement_score}</span>
        </div>
        <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
          <div
            className="h-full rounded-full transition-all duration-500"
            style={{
              width: `${person.social_engagement_score}%`,
              backgroundColor: color,
            }}
          />
        </div>
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-3 gap-1 text-xs font-mono text-gray-400">
        <div>
          <div className="text-gray-600">depth</div>
          <div>{person.estimated_depth}m</div>
        </div>
        <div>
          <div className="text-gray-600">orient</div>
          <div className="truncate text-[10px]">
          {person.body_orientation
            ? person.body_orientation.replace("_", " ").toLowerCase()
            : "unknown"}
        </div>
        </div>
        <div>
          <div className="text-gray-600">rank</div>
          <div>#{person.social_rank}</div>
        </div>
      </div>

      {/* Role reasoning */}
      {role && isSelected && (
        <div className="mt-2 pt-2 border-t border-gray-700 text-xs text-gray-400">
          {role.reasoning}
        </div>
      )}

      {/* Voice button — only shown when voice is enabled globally */}
      {voiceEnabled && (
        <button
          onClick={(e) => { e.stopPropagation(); onVoice(); }}
          disabled={isGeneratingVoice}
          className="mt-2 w-full text-xs py-1 rounded border border-gray-700 hover:border-green-500
                     text-gray-400 hover:text-green-400 transition-colors disabled:opacity-50"
        >
          {isGeneratingVoice && isSelected ? "🎙 Generating…" : "🔊 Generate Voice"}
        </button>
      )}
    </div>
  );
};
