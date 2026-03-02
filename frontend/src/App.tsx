import React, { useCallback, useRef, useState } from "react";
import { AnalysisResult, AppMode, DominanceResult } from "./types";
import { VideoCapture } from "./components/VideoCapture";
import { AnnotatedCanvas } from "./components/AnnotatedCanvas";
import { PersonCard } from "./components/PersonCard";
import { PersonDatabase } from "./components/PersonDatabase";
import { SocialGraph } from "./components/SocialGraph";
import { RankingPanel } from "./components/RankingPanel";
import { AudioPlayer } from "./components/AudioPlayer";
import { DominancePanel } from "./components/DominancePanel";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8080";

// ── WebSocket singleton ──────────────────────────────────────────────────────

let wsInstance: WebSocket | null = null;

function getWebSocket(
  onMessage: (data: AnalysisResult) => void,
  onError: (msg: string) => void,
): WebSocket {
  if (wsInstance && (wsInstance.readyState === WebSocket.OPEN || wsInstance.readyState === WebSocket.CONNECTING)) {
    return wsInstance;
  }
  const ws = new WebSocket(`${API_BASE.replace(/^http/, "ws")}/api/stream`);
  ws.binaryType = "arraybuffer";
  ws.onmessage = (e) => {
    try {
      const result: AnalysisResult = JSON.parse(e.data);
      if ("error" in result) onError((result as any).error);
      else onMessage(result);
    } catch {}
  };
  ws.onerror = () => onError("WebSocket error");
  ws.onclose = () => {
    if (wsInstance === ws) wsInstance = null;
  };
  wsInstance = ws;
  return ws;
}

function waitForOpen(ws: WebSocket, timeoutMs = 1200): Promise<boolean> {
  if (ws.readyState === WebSocket.OPEN) return Promise.resolve(true);
  if (ws.readyState !== WebSocket.CONNECTING) return Promise.resolve(false);
  return new Promise((resolve) => {
    const t = setTimeout(() => {
      ws.removeEventListener("open", onOpen);
      resolve(false);
    }, timeoutMs);
    const onOpen = () => {
      clearTimeout(t);
      resolve(true);
    };
    ws.addEventListener("open", onOpen, { once: true });
  });
}

// ─────────────────────────────────────────────────────────────────────────────

export default function App() {
  const [mode, setMode] = useState<AppMode>("snapshot");
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [selectedPersonId, setSelectedPersonId] = useState<string | null>(null);
  const [isGeneratingVoice, setIsGeneratingVoice] = useState(false);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [dialogue, setDialogue] = useState<string | null>(null);
  const [voiceProfile, setVoiceProfile] = useState<Record<string, string> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [processingMs, setProcessingMs] = useState<number | null>(null);
  const [voiceEnabled, setVoiceEnabled] = useState(false);
  const [pairMode, setPairMode] = useState(false);
  const [pairA, setPairA] = useState<string | null>(null);
  const [pairB, setPairB] = useState<string | null>(null);
  const [dominanceResult, setDominanceResult] = useState<DominanceResult | null>(null);
  const [isAnalyzingDominance, setIsAnalyzingDominance] = useState(false);
  const [faceDbPersons, setFaceDbPersons] = useState<string[]>([]);
  const [liveTargetA, setLiveTargetA] = useState<string | null>(null);
  const [liveTargetB, setLiveTargetB] = useState<string | null>(null);

  const lastLiveCall = useRef<number>(0);
  const lastDominanceMsRef = useRef<number>(0);

  // ── Analyze image via REST ─────────────────────────────────────────────────

  const analyzeBlob = useCallback(async (blob: Blob) => {
    setIsAnalyzing(true);
    setError(null);
    try {
      const form = new FormData();
      form.append("file", blob, "capture.jpg");
      const res = await fetch(`${API_BASE}/api/analyze`, {
        method: "POST",
        body: form,
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail ?? `HTTP ${res.status}`);
      }
      const data: AnalysisResult = await res.json();
      setResult(data);
      setSelectedPersonId(null);
      setProcessingMs(data.processing_time_ms);
    } catch (err: any) {
      setError(err.message || "Analysis failed");
    } finally {
      setIsAnalyzing(false);
    }
  }, []);

  // ── Live frame via WebSocket ───────────────────────────────────────────────

  const handleLiveFrame = useCallback(
    async (blob: Blob) => {
      const now = Date.now();
      if (now - lastLiveCall.current < 1000) return; // debounce
      lastLiveCall.current = now;

      const ws = getWebSocket(
        (data) => {
          setResult(data);
          setProcessingMs(data.processing_time_ms);
        },
        (msg) => setError(msg),
      );
      const ready = await waitForOpen(ws);
      if (!ready || ws.readyState !== WebSocket.OPEN) return;
      const arr = await blob.arrayBuffer();
      ws.send(arr);
    },
    [],
  );

  // ── Voice generation ───────────────────────────────────────────────────────

  const generateVoice = useCallback(
    async (personId: string) => {
      if (!result) return;
      setIsGeneratingVoice(true);
      setError(null);
      try {
        const res = await fetch(
          `${API_BASE}/api/person/${result.image_id}/${personId}/voice`,
          { method: "POST" },
        );
        if (!res.ok) {
          const data = await res.json().catch(() => ({}));
          throw new Error(data.detail ?? `HTTP ${res.status}`);
        }
        const profileHeader = res.headers.get("X-Voice-Profile");
        const dialogueHeader = res.headers.get("X-Dialogue");
        const audioBlob = await res.blob();
        const url = URL.createObjectURL(audioBlob);
        setAudioUrl(url);
        setDialogue(dialogueHeader);
        setVoiceProfile(profileHeader ? JSON.parse(profileHeader) : null);
      } catch (err: any) {
        setError(err.message || "Voice generation failed");
      } finally {
        setIsGeneratingVoice(false);
      }
    },
    [result],
  );

  const handlePersonClick = (id: string) => {
    if (pairMode) {
      if (pairA === id) { setPairA(null); return; }
      if (pairB === id) { setPairB(null); return; }
      if (!pairA) { setPairA(id); return; }
      if (!pairB) { setPairB(id); return; }
      // both slots full — replace B, shift old B out
      setPairA(pairB); setPairB(id);
    } else {
      setSelectedPersonId(id);
    }
  };

  const handlePersonVoice = (id: string) => {
    setSelectedPersonId(id);
    generateVoice(id);
  };

  const clearAudio = () => {
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    setAudioUrl(null);
    setDialogue(null);
    setVoiceProfile(null);
  };

  const exitPairMode = () => {
    setPairMode(false);
    setPairA(null);
    setPairB(null);
    setDominanceResult(null);
    setLiveTargetA(null);
    setLiveTargetB(null);
  };

  // Core dominance fetch — accepts explicit IDs so it can be called from both
  // manual button click (snapshot, uses LLM) and auto-trigger (live, CV-only).
  const analyzeDominanceForPair = useCallback(async (imageId: string, idA: string, idB: string, useLlm = true) => {
    setIsAnalyzingDominance(true);
    setError(null);
    try {
      const res = await fetch(
        `${API_BASE}/api/dominance/${imageId}/${idA}/${idB}?use_llm=${useLlm}`,
        { method: "POST" },
      );
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail ?? `HTTP ${res.status}`);
      }
      const data = await res.json();
      setDominanceResult({ ...data, person_id_a: idA, person_id_b: idB });
    } catch (err: any) {
      setError(err.message || "Dominance analysis failed");
    } finally {
      setIsAnalyzingDominance(false);
    }
  }, []);

  // Snapshot mode: triggered by button
  const analyzeDominance = async () => {
    if (!result || !pairA || !pairB) return;
    await analyzeDominanceForPair(result.image_id, pairA, pairB);
  };

  // ── Face DB list (for live pair pre-selection) ────────────────────────────

  React.useEffect(() => {
    if (!pairMode) return;
    fetch(`${API_BASE}/api/face-db`)
      .then((r) => r.json())
      .then((d) => setFaceDbPersons(d.persons ?? []))
      .catch(() => {});
  }, [pairMode]);

  // Reset debounce when live targets change so first detection triggers immediately
  React.useEffect(() => {
    lastDominanceMsRef.current = 0;
  }, [liveTargetA, liveTargetB]);

  // ── Auto-trigger dominance in live mode when both targets are detected ─────

  React.useEffect(() => {
    if (mode !== "live" || !pairMode || !liveTargetA || !liveTargetB || !result || isAnalyzingDominance) return;
    const personA = result.persons.find((p) => p.name === liveTargetA);
    const personB = result.persons.find((p) => p.name === liveTargetB);
    if (!personA || !personB) return;
    const now = Date.now();
    if (now - lastDominanceMsRef.current < 2000) return; // 2s debounce (CV-only is fast)
    lastDominanceMsRef.current = now;
    setPairA(personA.person_id);
    setPairB(personB.person_id);
    analyzeDominanceForPair(result.image_id, personA.person_id, personB.person_id, false); // no LLM in live mode
  }, [result, mode, pairMode, liveTargetA, liveTargetB, isAnalyzingDominance, analyzeDominanceForPair]);

  // ─────────────────────────────────────────────────────────────────────────

  const sortedPersons = result
    ? [...result.persons].sort((a, b) => a.social_rank - b.social_rank)
    : [];

  return (
    <div className="min-h-screen bg-[#0a0a0f] text-gray-100 font-sans">
      {/* ── Header ── */}
      <header className="border-b border-gray-800 px-6 py-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="text-green-400 font-mono font-bold text-xl">◈ SocialLens</span>
          <span className="text-xs text-gray-600 font-mono">
            CV + VLM Social Dynamics Analyzer
          </span>
        </div>

        <div className="flex items-center gap-2">
          {processingMs !== null && (
            <span className="text-xs font-mono text-gray-600">
              ⚡ {processingMs}ms
            </span>
          )}
          {/* Pair analysis toggle */}
          {result && result.persons.length >= 2 && (
            <button
              onClick={() => pairMode ? exitPairMode() : setPairMode(true)}
              className={`px-3 py-1 text-xs font-mono rounded-lg border transition-colors ${
                pairMode
                  ? "border-purple-600 bg-purple-500/20 text-purple-400"
                  : "border-gray-700 text-gray-500 hover:text-gray-300"
              }`}
              title="Analyze dominance between two people"
            >
              {pairMode ? "⚡ Pair ON" : "⚡ Pair"}
            </button>
          )}

          {/* Voice toggle */}
          <button
            onClick={() => setVoiceEnabled((v) => !v)}
            className={`px-3 py-1 text-xs font-mono rounded-lg border transition-colors ${
              voiceEnabled
                ? "border-green-600 bg-green-500/20 text-green-400"
                : "border-gray-700 text-gray-500 hover:text-gray-300"
            }`}
            title={voiceEnabled ? "Voice generation ON" : "Voice generation OFF"}
          >
            {voiceEnabled ? "🔊 Voice ON" : "🔇 Voice OFF"}
          </button>

          {/* Mode toggle */}
          <div className="flex rounded-lg border border-gray-700 overflow-hidden">
            {(["snapshot", "live"] as AppMode[]).map((m) => (
              <button
                key={m}
                onClick={() => setMode(m)}
                className={`px-3 py-1 text-xs font-mono transition-colors ${
                  mode === m
                    ? "bg-green-500/20 text-green-400"
                    : "text-gray-500 hover:text-gray-300"
                }`}
              >
                {m === "snapshot" ? "📷 Snapshot" : "📡 Live"}
              </button>
            ))}
          </div>
        </div>
      </header>

      {/* ── Main 3-column layout ── */}
      <div className="flex gap-4 p-4 h-[calc(100vh-57px)]">

        {/* Left column: Controls + Ranking */}
        <aside className="w-72 shrink-0 flex flex-col gap-4 overflow-y-auto">
          {/* Camera / upload */}
          <div className="glass-panel rounded-xl p-4 border border-gray-800">
            <div className="text-xs font-mono text-gray-500 uppercase mb-3">
              {mode === "live" ? "Live Feed" : "Image Input"}
            </div>
            <VideoCapture
              onSnapshot={analyzeBlob}
              onFileUpload={analyzeBlob}
              isAnalyzing={isAnalyzing}
              liveMode={mode === "live"}
              onLiveFrame={mode === "live" ? handleLiveFrame : undefined}
            />
          </div>

          {/* Error */}
          {error && (
            <div className="text-xs text-red-400 bg-red-900/20 border border-red-800 rounded-lg p-3 font-mono">
              ⚠ {error}
            </div>
          )}

          {/* Audio player */}
          {audioUrl && (
            <AudioPlayer
              audioUrl={audioUrl}
              dialogue={dialogue}
              voiceProfile={voiceProfile}
              personId={selectedPersonId}
              onClose={clearAudio}
            />
          )}

          {/* People Database */}
          <PersonDatabase />

          {/* Ranking */}
          {result && result.persons.length > 0 && (
            <div className="glass-panel rounded-xl p-4 border border-gray-800">
              <RankingPanel
                result={result}
                selectedPersonId={selectedPersonId}
                onPersonClick={handlePersonClick}
              />
            </div>
          )}
        </aside>

        {/* Center column: Annotated canvas */}
        <main className="flex-1 flex flex-col gap-3 min-w-0 min-h-0">
          {isAnalyzing && (
            <div className="text-center text-green-400 font-mono text-sm py-2 animate-pulse">
              ◈ Analyzing social dynamics…
            </div>
          )}
          <div className="flex-1 min-h-0">
            <AnnotatedCanvas
              result={result}
              onPersonClick={handlePersonClick}
              selectedPersonId={selectedPersonId}
              isGeneratingVoice={isGeneratingVoice}
            />
          </div>

          {/* Dynamics summary */}
          {result?.dynamics_summary && (
            <div className="glass-panel rounded-xl p-4 border border-gray-800 text-sm text-gray-300">
              <div className="text-xs font-mono text-gray-500 uppercase mb-1">AI Analysis</div>
              {result.dynamics_summary}
            </div>
          )}

          {/* Interesting observations */}
          {result?.interesting_observations?.length > 0 && (
            <div className="glass-panel rounded-xl p-4 border border-gray-800">
              <div className="text-xs font-mono text-gray-500 uppercase mb-2">Observations</div>
              <ul className="flex flex-col gap-1">
                {result.interesting_observations.map((obs, i) => (
                  <li key={i} className="text-xs text-gray-400 flex gap-2">
                    <span className="text-green-500 shrink-0">›</span>
                    {obs}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </main>

        {/* Right column: Person cards + graph */}
        <aside className="w-72 shrink-0 flex flex-col gap-4 overflow-y-auto">
          {/* Dominance panel */}
          {dominanceResult && (() => {
            const personA = result!.persons.find(p => p.person_id === dominanceResult.person_id_a);
            const personB = result!.persons.find(p => p.person_id === dominanceResult.person_id_b);
            return personA && personB ? (
              <DominancePanel
                result={dominanceResult}
                personA={personA}
                personB={personB}
                onClose={exitPairMode}
              />
            ) : null;
          })()}

          {/* Pair mode panel */}
          {pairMode && !dominanceResult && (
            <div className="glass-panel rounded-xl p-4 border border-purple-800/60 bg-purple-950/20">
              <div className="text-xs font-mono text-purple-400 uppercase mb-3">⚡ Pair Analysis</div>

              {mode === "live" ? (
                /* ── Live mode: pre-select from face DB ── */
                <>
                  <div className="text-xs text-gray-500 font-mono mb-3">
                    Select two people from your face database. Dominance will auto-analyze when both are detected.
                  </div>
                  {faceDbPersons.length === 0 ? (
                    <div className="text-xs text-gray-600 font-mono">No people in face database yet.</div>
                  ) : (
                    <>
                      {[
                        { label: "A", target: liveTargetA, setTarget: setLiveTargetA, color: "#ff4444" },
                        { label: "B", target: liveTargetB, setTarget: setLiveTargetB, color: "#00ffff" },
                      ].map(({ label, target, setTarget, color }) => {
                        const detected = target ? result?.persons.find((p) => p.name === target) : null;
                        return (
                          <div key={label} className="mb-3">
                            <div className="flex items-center gap-1.5 mb-1">
                              <span className="text-xs font-mono font-bold px-1 rounded border"
                                    style={{ color, borderColor: color }}>
                                {label}
                              </span>
                              {target && (
                                <span className={`text-[10px] font-mono ${detected ? "text-green-400" : "text-gray-600"}`}>
                                  {detected ? "● detected" : "○ not in frame"}
                                </span>
                              )}
                            </div>
                            <div className="flex flex-wrap gap-1">
                              {faceDbPersons.map((name) => (
                                <button
                                  key={name}
                                  onClick={() => setTarget(target === name ? null : name)}
                                  disabled={
                                    (label === "A" && liveTargetB === name) ||
                                    (label === "B" && liveTargetA === name)
                                  }
                                  className="text-xs px-2 py-0.5 rounded border transition-colors disabled:opacity-30"
                                  style={
                                    target === name
                                      ? { borderColor: color, color, backgroundColor: `${color}22` }
                                      : { borderColor: "#374151", color: "#9ca3af" }
                                  }
                                >
                                  {name}
                                </button>
                              ))}
                            </div>
                          </div>
                        );
                      })}
                      {isAnalyzingDominance && (
                        <div className="text-xs text-purple-400 font-mono animate-pulse">⚡ Analyzing…</div>
                      )}
                      {liveTargetA && liveTargetB && !isAnalyzingDominance && (
                        <div className="text-[10px] text-gray-600 font-mono mt-1">
                          CV-only · updates every 2s when both detected
                        </div>
                      )}
                    </>
                  )}
                </>
              ) : (
                /* ── Snapshot mode: click person cards ── */
                <>
                  <div className="text-xs text-gray-400 font-mono mb-3">
                    {!pairA && !pairB && "Click a person card to select Person A"}
                    {pairA && !pairB && "Now click another person to select Person B"}
                    {pairA && pairB && "Ready to analyze dominance"}
                  </div>
                  <button
                    onClick={analyzeDominance}
                    disabled={!pairA || !pairB || isAnalyzingDominance}
                    className="w-full text-xs py-1.5 rounded border border-purple-700 hover:border-purple-400
                               text-purple-400 hover:text-purple-300 transition-colors disabled:opacity-40"
                  >
                    {isAnalyzingDominance ? "⚡ Analyzing…" : "⚡ Analyze Dominance"}
                  </button>
                </>
              )}
            </div>
          )}

          {/* Social graph */}
          {result && result.persons.length > 1 && (
            <div className="glass-panel rounded-xl p-4 border border-gray-800">
              <SocialGraph
                result={result}
                selectedPersonId={selectedPersonId}
                onPersonClick={handlePersonClick}
              />
            </div>
          )}

          {/* Person cards */}
          {sortedPersons.map((p) => (
            <PersonCard
              key={p.person_id}
              person={p}
              result={result!}
              isSelected={!pairMode && selectedPersonId === p.person_id}
              isGeneratingVoice={isGeneratingVoice && selectedPersonId === p.person_id}
              voiceEnabled={voiceEnabled && !pairMode}
              pairRole={pairA === p.person_id ? "A" : pairB === p.person_id ? "B" : undefined}
              onClick={() => handlePersonClick(p.person_id)}
              onVoice={() => handlePersonVoice(p.person_id)}
            />
          ))}

          {!result && (
            <div className="glass-panel rounded-xl p-6 border border-gray-800 text-center text-gray-600 font-mono text-xs">
              Person analysis cards will appear here after scanning.
            </div>
          )}
        </aside>
      </div>
    </div>
  );
}
