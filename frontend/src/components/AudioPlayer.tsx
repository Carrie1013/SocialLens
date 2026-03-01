import React, { useEffect, useRef, useState } from "react";

interface Props {
  audioUrl: string | null;
  dialogue: string | null;
  voiceProfile: Record<string, string> | null;
  personId: string | null;
  onClose: () => void;
}

export const AudioPlayer: React.FC<Props> = ({
  audioUrl,
  dialogue,
  voiceProfile,
  personId,
  onClose,
}) => {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [playing, setPlaying] = useState(false);
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    if (audioUrl && audioRef.current) {
      audioRef.current.src = audioUrl;
      audioRef.current.play().catch(() => {});
      setPlaying(true);
      setProgress(0);
    }
  }, [audioUrl]);

  const handleTimeUpdate = () => {
    const el = audioRef.current;
    if (!el || !el.duration) return;
    setProgress((el.currentTime / el.duration) * 100);
  };

  const handleEnded = () => {
    setPlaying(false);
    setProgress(100);
  };

  const togglePlay = () => {
    const el = audioRef.current;
    if (!el) return;
    if (playing) {
      el.pause();
      setPlaying(false);
    } else {
      el.play().catch(() => {});
      setPlaying(true);
    }
  };

  if (!audioUrl) return null;

  return (
    <div className="glass-panel rounded-xl p-4 border border-green-500/30">
      <audio
        ref={audioRef}
        onTimeUpdate={handleTimeUpdate}
        onEnded={handleEnded}
        className="hidden"
      />

      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="font-mono text-sm text-green-400">
          🎙 Voice: {personId}
        </div>
        <button
          onClick={onClose}
          className="text-gray-500 hover:text-gray-300 text-lg leading-none"
        >
          ×
        </button>
      </div>

      {/* Dialogue */}
      {dialogue && (
        <div className="text-sm text-gray-300 italic mb-3 border-l-2 border-green-500/50 pl-3">
          "{dialogue}"
        </div>
      )}

      {/* Voice profile badges */}
      {voiceProfile && (
        <div className="flex flex-wrap gap-1 mb-3">
          {Object.entries(voiceProfile)
            .filter(([k]) => k !== "sample_dialogue")
            .map(([k, v]) => (
              <span
                key={k}
                className="text-xs bg-gray-800 text-gray-400 px-2 py-0.5 rounded-full border border-gray-700"
              >
                {k.replace(/_/g, " ")}: <span className="text-gray-200">{v}</span>
              </span>
            ))}
        </div>
      )}

      {/* Controls */}
      <div className="flex items-center gap-3">
        <button
          onClick={togglePlay}
          className="w-9 h-9 rounded-full bg-green-500 text-black flex items-center justify-center
                     hover:bg-green-400 transition-colors text-sm font-bold shrink-0"
        >
          {playing ? "⏸" : "▶"}
        </button>

        <div className="flex-1 h-1.5 bg-gray-800 rounded-full overflow-hidden">
          <div
            className="h-full bg-green-500 rounded-full transition-all"
            style={{ width: `${progress}%` }}
          />
        </div>
      </div>
    </div>
  );
};
