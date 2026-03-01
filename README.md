# SocialLens — Real-Time Social Dynamics Visualizer

A full-stack portfolio project combining **computer vision**, **vision-language models**, and **text-to-speech** to analyze social dynamics from images and webcam feeds in real time.

```
[Webcam / Image Upload]
        ↓
  CV Pipeline (MediaPipe)          ← fast, runs in <200ms
  ├── Face detection & bounding boxes
  ├── Facial landmark expression analysis
  ├── Pose-based body orientation
  ├── Depth estimation (face-area proxy)
  └── Pairwise social distance scores
        ↓
  VLM Analysis (Claude Sonnet)     ← async, cached 3s
  ├── Group detection
  ├── Social role assignment (leader / speaker / listener …)
  ├── Scene dynamics summary
  └── Interesting observations
        ↓
  ElevenLabs TTS (on click)        ← per-person voice generation
  ├── Character voice profiling via Claude VLM
  └── Synthesised MP3 audio
        ↓
  React Frontend
  ├── Annotated canvas with overlays
  ├── Social graph visualisation
  ├── Ranked person cards
  └── Audio player
```

---

## Tech Stack

| Layer     | Technology                                     |
|-----------|------------------------------------------------|
| Backend   | Python 3.11+, FastAPI, WebSocket (uvicorn)     |
| CV        | OpenCV, MediaPipe (face mesh + pose)           |
| AI        | Anthropic Claude `claude-sonnet-4-20250514`    |
| TTS       | ElevenLabs `eleven_multilingual_v2`            |
| Frontend  | React 18, TypeScript, Vite, Tailwind CSS       |

---

## Project Structure

```
sociallens/
├── backend/
│   ├── main.py              # FastAPI app + WebSocket endpoint
│   ├── cv_pipeline.py       # MediaPipe face/pose detection & analysis
│   ├── vlm_analyzer.py      # Claude VLM integration (scene + voice profile)
│   ├── elevenlabs_client.py # ElevenLabs TTS with voice mapping
│   ├── social_metrics.py    # Pairwise social distance & engagement scores
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.tsx                    # Root component, API wiring
│   │   ├── components/
│   │   │   ├── VideoCapture.tsx       # Webcam + file upload
│   │   │   ├── AnnotatedCanvas.tsx    # Overlay renderer (groups, roles, connections)
│   │   │   ├── PersonCard.tsx         # Per-person stats card with voice button
│   │   │   ├── SocialGraph.tsx        # Force-graph-style circular network view
│   │   │   ├── RankingPanel.tsx       # Ranked leaderboard
│   │   │   └── AudioPlayer.tsx        # ElevenLabs audio playback
│   │   ├── types/index.ts             # Shared TypeScript types
│   │   └── index.css                  # Cyberpunk dark theme
│   ├── package.json
│   ├── vite.config.ts
│   └── index.html
├── .env.example
├── .gitignore
└── README.md
```

---

## Setup

### 1 — API Keys

```bash
cp .env.example .env
# Edit .env and fill in your keys
```

| Key                   | Required | Where to get                        |
|-----------------------|----------|-------------------------------------|
| `ANTHROPIC_API_KEY`   | Yes      | https://console.anthropic.com/      |
| `ELEVENLABS_API_KEY`  | Optional | https://elevenlabs.io/              |

### 2 — Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **Note on MediaPipe**: MediaPipe requires Python 3.8–3.12. It does **not** yet support 3.13.

### 3 — Frontend

```bash
cd frontend
npm install
```

---

## Running

### Start the backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### Start the frontend

```bash
cd frontend
npm run dev
```

Open `http://localhost:5173` in your browser.

---

## API Endpoints

| Method | Path                                       | Description                                |
|--------|--------------------------------------------|--------------------------------------------|
| GET    | `/api/health`                              | Service health + feature flags             |
| POST   | `/api/analyze`                             | Upload image → full pipeline result        |
| POST   | `/api/person/{image_id}/{person_id}/voice` | Generate TTS voice for a detected person   |
| WS     | `/api/stream`                              | WebSocket: send JPEG bytes → JSON result   |

### `/api/analyze` response shape

```json
{
  "image_id": "uuid",
  "persons": [
    {
      "person_id": "P1",
      "bbox": [x, y, w, h],
      "estimated_depth": 3.2,
      "body_orientation": "FACING_CAMERA",
      "expression": "SMILING",
      "face_area_px": 8420,
      "social_engagement_score": 78.5,
      "social_rank": 1
    }
  ],
  "groups": [...],
  "roles": { "P1": { "role": "leader", "confidence": 85, "reasoning": "..." } },
  "social_center": "P1",
  "dynamics_summary": "Two people appear to be in an animated conversation...",
  "interesting_observations": ["...", "..."],
  "social_ranking": ["P1", "P3", "P2"],
  "annotated_image": "<base64 JPEG>",
  "processing_time_ms": 450
}
```

---

## Features

### Snapshot Mode
Upload an image or click **Snapshot & Analyze** to capture a webcam frame. The full CV + VLM pipeline runs and results are displayed immediately.

### Live Mode
Switch to **Live** mode — the app automatically captures and analyzes a frame every 2 seconds, updating the overlay in real time.

### Person Voice Generation
Click any person bounding box or the **Generate Voice** button on a person card. Claude analyzes the person's appearance and generates a character profile (age, personality, speaking style), then ElevenLabs synthesises a sample line of dialogue in a matching voice.

### Social Annotations
- **Coloured boxes**: Green = high engagement, amber = medium, red = isolated
- **Group outlines**: Semi-transparent regions around detected social groups
- **Connection lines**: Animated links between group members
- **Role badges**: Crown (leader), microphone (speaker), ear (listener), eye (observer), chain (connector)
- **Social center crown**: The most socially central person is marked

---

## Architecture Notes

- **CV Pipeline** runs synchronously on CPU in ~50–150ms using MediaPipe.
- **VLM analysis** is async and cached with a 3-second TTL to avoid redundant API calls during live mode.
- **Depth estimation** uses face bounding box area as a proxy — larger face = closer to camera. This is a heuristic and works best when people are at similar heights.
- **Multi-person pose** — MediaPipe Pose detects one body. For groups, orientation is estimated from the overall scene pose. Full multi-person pose would require a heavier model.
- **Voice IDs** in `elevenlabs_client.py` are from ElevenLabs' premade voice library. You can replace them with cloned or custom voices from your ElevenLabs account.

---

## Demo

> *Add a demo GIF here once you have a recording.*

![Demo placeholder](https://via.placeholder.com/800x450/0a0a0f/00ff88?text=SocialLens+Demo)

---

## License

MIT
