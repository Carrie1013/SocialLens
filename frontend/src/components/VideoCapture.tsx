import React, { useCallback, useRef, useState } from "react";

interface Props {
  onSnapshot: (blob: Blob) => void;
  onFileUpload: (blob: Blob) => void;
  isAnalyzing: boolean;
  liveMode: boolean;
  onLiveFrame?: (blob: Blob) => void;
}

const LIVE_INTERVAL_MS = 2000;

export const VideoCapture: React.FC<Props> = ({
  onSnapshot,
  onFileUpload,
  isAnalyzing,
  liveMode,
  onLiveFrame,
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const liveTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const liveTickBusyRef = useRef(false);

  const [cameraActive, setCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);

  // ── Camera helpers ──────────────────────────────────────────────────────────

  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720, facingMode: "user" },
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        setCameraActive(true);
        setCameraError(null);
      }
    } catch (err: any) {
      setCameraError(err.message || "Camera access denied");
    }
  }, []);

  const stopCamera = useCallback(() => {
    if (videoRef.current?.srcObject) {
      (videoRef.current.srcObject as MediaStream)
        .getTracks()
        .forEach((t) => t.stop());
      videoRef.current.srcObject = null;
    }
    setCameraActive(false);
    if (liveTimerRef.current) {
      clearInterval(liveTimerRef.current);
      liveTimerRef.current = null;
    }
  }, []);

  const captureFrame = useCallback(async (): Promise<Blob | null> => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return null;
    canvas.width = video.videoWidth || 1280;
    canvas.height = video.videoHeight || 720;
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    ctx.drawImage(video, 0, 0);
    return await new Promise((resolve) => {
      canvas.toBlob((b) => resolve(b), "image/jpeg", 0.85);
    });
  }, []);

  // ── Snapshot ────────────────────────────────────────────────────────────────

  const handleSnapshot = useCallback(async () => {
    const blob = await captureFrame();
    if (blob) onSnapshot(blob);
  }, [captureFrame, onSnapshot]);

  // ── Live mode ───────────────────────────────────────────────────────────────

  React.useEffect(() => {
    if (liveMode && cameraActive && onLiveFrame) {
      liveTimerRef.current = setInterval(async () => {
        if (liveTickBusyRef.current) return;
        liveTickBusyRef.current = true;
        try {
          const blob = await captureFrame();
          if (blob) onLiveFrame(blob);
        } finally {
          liveTickBusyRef.current = false;
        }
      }, LIVE_INTERVAL_MS);
    } else {
      if (liveTimerRef.current) {
        clearInterval(liveTimerRef.current);
        liveTimerRef.current = null;
      }
      liveTickBusyRef.current = false;
    }
    return () => {
      if (liveTimerRef.current) clearInterval(liveTimerRef.current);
      liveTickBusyRef.current = false;
    };
  }, [liveMode, cameraActive, captureFrame, onLiveFrame]);

  // ── File upload ─────────────────────────────────────────────────────────────

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) onFileUpload(file);
    e.target.value = "";
  };

  // ── Render ──────────────────────────────────────────────────────────────────

  return (
    <div className="flex flex-col gap-3">
      {/* Hidden capture canvas */}
      <canvas ref={canvasRef} className="hidden" />

      {/* Video preview */}
      <div className="relative bg-black rounded-lg overflow-hidden border border-gray-700 aspect-video">
        <video
          ref={videoRef}
          className="w-full h-full object-cover"
          muted
          playsInline
        />
        {!cameraActive && (
          <div className="absolute inset-0 flex items-center justify-center text-gray-500">
            <div className="text-center">
              <div className="text-4xl mb-2">📷</div>
              <div className="text-sm font-mono">No camera feed</div>
            </div>
          </div>
        )}
        {cameraActive && (
          <div className="absolute top-2 right-2 flex items-center gap-1">
            <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
            <span className="text-xs text-red-400 font-mono">LIVE</span>
          </div>
        )}
        {liveMode && cameraActive && (
          <div className="absolute bottom-2 left-2 text-xs font-mono text-green-400">
            AUTO-SCAN: every {LIVE_INTERVAL_MS / 1000}s
          </div>
        )}
      </div>

      {/* Error */}
      {cameraError && (
        <div className="text-xs text-red-400 font-mono bg-red-900/20 rounded p-2">
          ⚠ {cameraError}
        </div>
      )}

      {/* Controls */}
      <div className="flex gap-2 flex-wrap">
        {!cameraActive ? (
          <button
            onClick={startCamera}
            className="btn-primary flex-1"
          >
            Start Camera
          </button>
        ) : (
          <>
            <button
              onClick={handleSnapshot}
              disabled={isAnalyzing}
              className="btn-primary flex-1"
            >
              {isAnalyzing ? "Analyzing…" : "Snapshot & Analyze"}
            </button>
            <button
              onClick={stopCamera}
              className="btn-secondary"
            >
              Stop
            </button>
          </>
        )}

        <button
          onClick={() => fileInputRef.current?.click()}
          disabled={isAnalyzing}
          className="btn-secondary flex-1"
        >
          Upload Image
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={handleFileChange}
        />
      </div>
    </div>
  );
};
