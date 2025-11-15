import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import List, Optional

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from groq import Groq
from pydantic import BaseModel, Field

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("meetlite")

FASTINO_BASE_URL = os.getenv("FASTINO_BASE_URL", "https://api.fastino.ai")
FASTINO_API_KEY = os.getenv("FASTINO_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
FREEPIK_API_KEY = os.getenv("FREEPIK_API_KEY")
DAYTONA_API_KEY = os.getenv("DAYTONA_API_KEY")
DAYTONA_BASE_URL = os.getenv("DAYTONA_BASE_URL")
DAYTONA_TARGET = os.getenv("DAYTONA_TARGET")

REQUEST_TIMEOUT = 30
http_session = requests.Session()
_groq_client: Optional[Groq] = None

app = FastAPI(title="MeetLite")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)


class AnalyzeRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    transcript: str = Field(..., min_length=1)


class ActionItem(BaseModel):
    description: str
    owner: Optional[str] = None
    due_date: Optional[str] = None
    priority: Optional[str] = None


class VisualResponse(BaseModel):
    image_url: Optional[str]
    query_used: str


class AnalyzeResponse(BaseModel):
    summary: str
    action_items: List[ActionItem]
    visual: VisualResponse


def _require_env(value: Optional[str], name: str) -> str:
    if not value:
        raise HTTPException(status_code=500, detail=f"Missing required environment variable: {name}")
    return value


def _fastino_headers() -> dict:
    api_key = _require_env(FASTINO_API_KEY, "FASTINO_API_KEY")
    return {
        "x-api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        api_key = _require_env(GROQ_API_KEY, "GROQ_API_KEY")
        _groq_client = Groq(api_key=api_key)
    return _groq_client


def simple_clean_transcript(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


PREPROCESS_SCRIPT = """import re
from pathlib import Path

raw = Path('transcript.txt').read_text(encoding='utf-8', errors='ignore')
raw = re.sub(r'[\U0001F1E0-\U0001FADB]+', ' ', raw)
raw = re.sub(r'[\u2600-\u27BF]+', ' ', raw)
raw = raw.replace('\r', '\n')
raw = re.sub(r'\n+', '\n', raw)
raw = re.sub(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', '[email]', raw)
raw = re.sub(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', '[phone]', raw)
raw = re.sub(r'[^\x09\x0A\x0D\x20-\x7E]', ' ', raw)
raw = re.sub(r'\s+', ' ', raw)
print(raw.strip())
"""


def _get_daytona_client():
    try:
        from daytona import Daytona, DaytonaConfig  # type: ignore
    except Exception as exc:  # pragma: no cover - import issues surfaced at runtime
        raise HTTPException(status_code=500, detail=f"Daytona SDK unavailable: {exc}") from exc

    api_key = DAYTONA_API_KEY or os.getenv("DAYTONA_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Missing Daytona API key.")

    cfg_kwargs = {"api_key": api_key}
    api_url = DAYTONA_BASE_URL or os.getenv("DAYTONA_API_URL")
    if api_url:
        cfg_kwargs["api_url"] = api_url
    target = DAYTONA_TARGET or os.getenv("DAYTONA_TARGET")
    if target:
        cfg_kwargs["target"] = target

    config = DaytonaConfig(**cfg_kwargs)
    return Daytona(config)


def _daytona_preprocess_code(transcript: str) -> str:
    transcript_literal = json.dumps(transcript)
    script_literal = json.dumps(PREPROCESS_SCRIPT)
    return (
        "from pathlib import Path\n"
        "import subprocess, sys\n"
        f"Path('transcript.txt').write_text({transcript_literal}, encoding='utf-8', errors='ignore')\n"
        f"Path('preprocess.py').write_text({script_literal}, encoding='utf-8')\n"
        "proc = subprocess.run([sys.executable, 'preprocess.py'], capture_output=True, text=True, timeout=60)\n"
        "if proc.stdout:\n"
        "    print(proc.stdout)\n"
        "elif proc.stderr:\n"
        "    print(proc.stderr)\n"
    )


def daytona_preprocess_transcript(user_id: str, transcript: str) -> str:
    if not transcript.strip():
        return ""
    try:
        client = _get_daytona_client()
    except HTTPException as exc:
        logger.warning("Daytona client unavailable (%s), using fallback cleaning.", exc.detail)
        return simple_clean_transcript(transcript)

    sandbox = None
    try:
        sandbox = client.create()
        script = _daytona_preprocess_code(transcript)
        response = sandbox.process.code_run(script)
        output = getattr(response, "result", None) or getattr(response, "stdout", "") or ""
        cleaned = output.strip()
        if not cleaned:
            return simple_clean_transcript(transcript)
        return cleaned
    except Exception as exc:  # noqa: BLE001
        logger.exception("Daytona preprocessing failed, using fallback: %s", exc)
        return simple_clean_transcript(transcript)
    finally:
        if sandbox:
            try:
                sandbox.delete()
            except Exception:  # noqa: BLE001
                logger.warning("Failed to delete Daytona sandbox")


def fastino_upsert_user(user_identifier: str) -> str:
    email = user_identifier if "@" in user_identifier else f"{user_identifier}@meetlite.local"
    payload = {
        "email": email,
        "purpose": "MeetLite meeting assistant personalization",
        "traits": {
            "source": "meetlite",
            "preferred_medium": "meetings",
        },
    }
    response = http_session.post(
        f"{FASTINO_BASE_URL.rstrip('/')}/register",
        headers=_fastino_headers(),
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()
    fastino_user_id = data.get("user_id")
    if not fastino_user_id:
        raise HTTPException(status_code=502, detail="Fastino registration did not return user_id")
    return fastino_user_id


def fastino_ingest(user_id: str, clean_transcript: str) -> None:
    payload = {
        "user_id": user_id,
        "source": "meetlite",
        "documents": [
            {
                "content": clean_transcript,
                "title": "Meeting transcript",
                "document_type": "message_history",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        ],
        "options": {"dedupe": True},
    }
    response = http_session.post(
        f"{FASTINO_BASE_URL.rstrip('/')}/ingest",
        headers=_fastino_headers(),
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()


def fastino_extract_action_items(clean_transcript: str) -> List[ActionItem]:
    payload = {
        "task": "extract_json",
        "text": clean_transcript,
        "threshold": 0.3,
        "schema": {
            "action_item": [
                "description::str::The concrete action item someone agreed to do",
                "owner::str::Person responsible, if mentioned",
                "due_date::str::Due date or time phrase",
                "priority::str::Priority like high, medium, or low, if implied",
            ]
        },
    }
    response = http_session.post(
        f"{FASTINO_BASE_URL.rstrip('/')}/gliner-2",
        headers=_fastino_headers(),
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()
    raw_items = data.get("result", {}).get("action_item", [])
    normalized: List[ActionItem] = []
    for item in raw_items:
        description = item.get("description") or item.get("value") or ""
        if not description:
            continue
        normalized.append(
            ActionItem(
                description=description.strip(),
                owner=(item.get("owner") or None),
                due_date=(item.get("due_date") or None),
                priority=(item.get("priority") or None),
            )
        )
    return normalized


def fastino_get_user_context(user_id: str) -> str:
    summary_text = ""
    try:
        response = http_session.get(
            f"{FASTINO_BASE_URL.rstrip('/')}/summary",
            headers=_fastino_headers(),
            params={"user_id": user_id, "max_chars": 800},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        summary_text = response.json().get("summary", "")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Fastino summary lookup failed: %s", exc)

    snippets: List[str] = []
    try:
        response = http_session.post(
            f"{FASTINO_BASE_URL.rstrip('/')}/chunks",
            headers=_fastino_headers(),
            json={
                "user_id": user_id,
                "history": [
                    {"role": "system", "content": "You summarize meetings for the user."},
                    {
                        "role": "user",
                        "content": "Context request: highlight similar past meetings or commitments.",
                    },
                ],
                "k": 3,
                "similarity_threshold": 0.3,
            },
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        raw_chunks = data.get("chunks") or data.get("result") or []
        for chunk in raw_chunks:
            text = chunk.get("text") if isinstance(chunk, dict) else chunk
            if text:
                snippets.append(str(text).strip())
    except Exception as exc:  # noqa: BLE001
        logger.warning("Fastino chunk retrieval failed: %s", exc)

    parts = ["User history from Fastino:"]
    if summary_text:
        parts.append(summary_text.strip())
    else:
        parts.append("No stored summary.")
    if snippets:
        parts.append("Relevant past snippets:")
        for snippet in snippets:
            parts.append(f"- {snippet}")
    return "\n".join(parts)


def groq_generate_summary(user_context: str, clean_transcript: str, action_items: List[ActionItem]) -> str:
    client = _get_groq_client()
    system_prompt = (
        "You are a concise meeting summarizer.\n"
        "Use the user's historical context and style hints below, coming from Fastino's personalization memory.\n"
        "User Context:\n{user_context}\n\n"
        "Given the new transcript and the extracted action_items, produce 3-7 bullet points summarizing the key decisions, "
        "topics, and follow-ups.\nBe clear, factual, and concise."
    )
    payload = {
        "transcript": clean_transcript,
        "action_items": [item.dict() for item in action_items],
    }
    completion = client.chat.completions.create(
        model="qwen/qwen3-32b",
        messages=[
            {"role": "system", "content": system_prompt.format(user_context=user_context or "No user history available.")},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        temperature=0.2,
        max_tokens=512,
    )
    if not completion.choices:
        raise HTTPException(status_code=502, detail="Groq summary generation failed")
    return completion.choices[0].message.content.strip()


def groq_generate_freepik_query(summary: str) -> str:
    client = _get_groq_client()
    completion = client.chat.completions.create(
        model="qwen/qwen3-32b",
        messages=[
            {
                "role": "system",
                "content": "You are an assistant that generates short Freepik search queries.",
            },
            {
                "role": "user",
                "content": (
                    "Given this meeting summary, generate a short Freepik search query for an illustration "
                    "that could visually represent the meeting:\n"
                    f"{summary}"
                ),
            },
        ],
        temperature=0.2,
        max_tokens=64,
    )
    if not completion.choices:
        raise HTTPException(status_code=502, detail="Groq query generation failed")
    query = completion.choices[0].message.content.strip()
    return query.strip('"').strip()


def freepik_search_image(query: str) -> Optional[str]:
    if not query:
        return None
    api_key = _require_env(FREEPIK_API_KEY, "FREEPIK_API_KEY")
    params = {"query": query, "limit": 1, "order": "popular"}
    response = http_session.get(
        "https://api.freepik.com/v1/resources",
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
        params=params,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()
    items = data.get("data") or data.get("items") or []
    if not items:
        return None
    first = items[0]
    for key in ("preview", "preview_url", "image", "url"):
        if key in first and first[key]:
            return first[key]
    return None


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(payload: AnalyzeRequest) -> AnalyzeResponse:
    clean_transcript = daytona_preprocess_transcript(payload.user_id, payload.transcript)
    fastino_user_id = fastino_upsert_user(payload.user_id)
    fastino_ingest(fastino_user_id, clean_transcript)
    action_items = fastino_extract_action_items(clean_transcript)
    user_context = fastino_get_user_context(fastino_user_id)
    summary_text = groq_generate_summary(user_context, clean_transcript, action_items)
    try:
        query = groq_generate_freepik_query(summary_text)
    except HTTPException:
        query = ""
    image_url = None
    if query:
        try:
            image_url = freepik_search_image(query)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Freepik search failed: %s", exc)
            image_url = None
    visual = VisualResponse(image_url=image_url, query_used=query or "")
    return AnalyzeResponse(summary=summary_text, action_items=action_items, visual=visual)


app.mount("/", StaticFiles(directory=".", html=True), name="static")
