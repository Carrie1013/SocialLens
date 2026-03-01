# 1. Set up keys
cp .env.example .env   # fill in ANTHROPIC_API_KEY + ELEVENLABS_API_KEY

# 2. Backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8080

# 3. Frontend (new terminal)
cd frontend
npm install
npm run dev
# → http://localhost:3000