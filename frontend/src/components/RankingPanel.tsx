import React from "react";
import {
  AnalysisResult,
  PersonData,
  EXPRESSION_EMOJI,
  ROLE_ICON,
  SCORE_COLOR,
} from "../types";

interface Props {
  result: AnalysisResult;
  selectedPersonId: string | null;
  onPersonClick: (id: string) => void;
}

const RANK_MEDAL: Record<number, string> = { 1: "🥇", 2: "🥈", 3: "🥉" };

export const RankingPanel: React.FC<Props> = ({
  result,
  selectedPersonId,
  onPersonClick,
}) => {
  // Sort by rank
  const ranked = [...result.persons].sort((a, b) => a.social_rank - b.social_rank);

  return (
    <div>
      <div className="text-xs font-mono text-gray-500 uppercase mb-3">Social Ranking</div>
      <div className="flex flex-col gap-2">
        {ranked.map((p) => {
          const color = SCORE_COLOR(p.social_engagement_score);
          const role = result.roles?.[p.person_id];
          const isSelected = p.person_id === selectedPersonId;
          const medal = RANK_MEDAL[p.social_rank] ?? `#${p.social_rank}`;

          return (
            <div
              key={p.person_id}
              onClick={() => onPersonClick(p.person_id)}
              className={`
                flex items-center gap-3 p-2 rounded-lg cursor-pointer border transition-all
                ${isSelected ? "border-white/40 bg-white/5" : "border-gray-800 hover:border-gray-600"}
              `}
            >
              {/* Rank medal */}
              <div className="text-lg w-7 text-center shrink-0">{medal}</div>

              {/* Person ID + role */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-1 mb-0.5">
                  <span className="font-mono font-bold text-sm" style={{ color }}>
                    {p.person_id}
                  </span>
                  {result.social_center === p.person_id && (
                    <span className="text-xs text-yellow-400">center</span>
                  )}
                  {role && (
                    <span className="text-xs text-gray-500">
                      {ROLE_ICON[role.role]}{role.role}
                    </span>
                  )}
                </div>

                {/* Score bar */}
                <div className="flex items-center gap-2">
                  <div className="flex-1 h-1 bg-gray-800 rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full"
                      style={{
                        width: `${p.social_engagement_score}%`,
                        backgroundColor: color,
                      }}
                    />
                  </div>
                  <span className="text-xs font-mono shrink-0" style={{ color }}>
                    {p.social_engagement_score}
                  </span>
                </div>
              </div>

              {/* Expression */}
              <div className="text-xl shrink-0">
                {EXPRESSION_EMOJI[p.expression]}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
