// ─── Core data types ─────────────────────────────────────────────────────────

export type BodyOrientation =
  | "FACING_CAMERA"
  | "TURNED_AWAY"
  | "PROFILE_LEFT"
  | "PROFILE_RIGHT"
  | "ANGLED"
  | "UNKNOWN";

export type Expression =
  | "NEUTRAL"
  | "SMILING"
  | "TALKING"
  | "SURPRISED"
  | "FOCUSED"
  | "LAUGHING"
  | "UNKNOWN";

export type SocialRole =
  | "leader"
  | "listener"
  | "speaker"
  | "observer"
  | "connector";

export type GroupType = "conversation" | "presentation" | "casual";

export interface PersonData {
  person_id: string;
  name?: string;             // Set when face is matched in personal database
  bbox: [number, number, number, number]; // [x, y, w, h]
  estimated_depth: number;
  body_orientation: BodyOrientation;
  expression: Expression;
  face_area_px: number;
  social_engagement_score: number;
  social_rank: number;
  landmark_data?: {
    source?: string;
    conf?: number;
    body_focus_vector?: [number, number] | null;
    head_gaze_vector?: [number, number] | null;
    head_gaze_source?: string;
  };
}

export interface PersonRole {
  role: SocialRole;
  confidence: number;
  reasoning: string;
}

export interface BoundingRegion {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface GroupData {
  group_id: string;
  member_ids: string[];
  group_type: GroupType;
  cohesion_score: number;
  bounding_region: BoundingRegion;
}

export interface AnalysisResult {
  image_id: string;
  persons: PersonData[];
  groups: GroupData[];
  roles: Record<string, PersonRole>;
  social_center: string | null;
  dynamics_summary: string;
  interesting_observations: string[];
  social_ranking: string[];
  annotated_image: string; // base64 JPEG
  processing_time_ms: number;
}

export interface VoiceProfile {
  age_estimate: string;
  gender_presentation: "masculine" | "feminine" | "androgynous";
  personality_impression: string;
  accent_suggestion: string;
  speaking_style: string;
  sample_dialogue: string;
}

// ─── UI state types ───────────────────────────────────────────────────────────

export type AppMode = "snapshot" | "live";

export interface UIState {
  mode: AppMode;
  isAnalyzing: boolean;
  selectedPersonId: string | null;
  isGeneratingVoice: boolean;
  error: string | null;
}

// ─── Constants ────────────────────────────────────────────────────────────────

export const EXPRESSION_EMOJI: Record<Expression, string> = {
  NEUTRAL: "😐",
  SMILING: "😊",
  TALKING: "💬",
  SURPRISED: "😮",
  FOCUSED: "🧐",
  LAUGHING: "😄",
  UNKNOWN: "❓",
};

export const ROLE_ICON: Record<SocialRole, string> = {
  leader: "👑",
  listener: "👂",
  speaker: "🎤",
  observer: "👁",
  connector: "🔗",
};

export const SCORE_COLOR = (score: number): string => {
  if (score >= 60) return "#00ff88";  // neon green
  if (score >= 35) return "#ffaa00";  // amber
  return "#ff4444";                   // red
};
