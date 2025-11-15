# MeetLite

MeetLite is a minimal FastAPI app that turns raw meeting transcripts into concise summaries, action items, and a Freepik-powered visual recap. It relies entirely on Fastino for personalization and Daytona for safe, local transcript pre-processing.

## Features

- Daytona sandbox cleans transcripts locally (whitespace normalization, emoji stripping, optional PII masking).
- Fastino powers personalization: user upsert, ingestion, GLiNER-2 action item extraction, summary + chunk retrieval.
- Groq LLM (`llama-3.1-70b-versatile`) produces the personalized bullet summary; a lighter Groq call crafts the Freepik search query.
- Freepik API delivers a relevant illustration.
- Single-page frontend (`index.html`) interacts with the `/analyze` endpoint and renders the response.

## Setup

1. **Create a virtual environment** (Python 3.10+ recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Export required environment variables**:

   ```bash
   export FASTINO_API_KEY="<your_fastino_api_key>"
   export GROQ_API_KEY="<your_groq_api_key>"
   export FREEPIK_API_KEY="<your_freepik_api_key>"
   export DAYTONA_API_KEY="<your_daytona_api_key>"
   export DAYTONA_BASE_URL="<your_daytona_server_url>"
   # Optional override if Fastino exposes a different hostname
   # export FASTINO_BASE_URL="https://api.fastino.ai"
   ```

4. **Run the dev server**:

   ```bash
   uvicorn main:app --reload
   ```

5. **Open the UI**: visit [http://localhost:8000/](http://localhost:8000/). Paste a transcript, provide a user identifier, and click **Analyze Meeting**.

## Daytona Workflow

`daytona_preprocess_transcript` creates a sandbox, uploads `preprocess.py` plus the raw transcript, runs the script locally (no outbound HTTP from inside the sandbox), and reads stdout as the cleaned transcript. If Daytona is unavailable, MeetLite falls back to a safe in-process whitespace normalizer so the flow still works end-to-end.

## Fastino & Groq Integration

1. Upsert the user with Fastino `/users`.
2. `/ingest` stores the cleaned transcript with metadata.
3. `/gliner-2` extracts structured action items.
4. `/summary` + `/retrieve-chunks` supply personalization context fed into Groq.
5. Groq summarizes + crafts a Freepik search query, which is then resolved via the Freepik API to display a recap image.

## Notes

- No local or custom memory is stored—only Fastino maintains personalization state.
- Daytona is never used for network calls; it only executes the local `preprocess.py` helper.
- All outbound HTTP interactions (Fastino, Groq, Freepik) are initiated from the FastAPI server process.
