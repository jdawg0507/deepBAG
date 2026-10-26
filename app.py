import os
import sys
import json
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# Load .env file automatically
load_dotenv()

# Add directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))



from nba_nlp_analyzer_modular import NBAQueryAnalyzer
from core.schemas import ChatRequest, ChatResponse
from core.telemetry import TelemetryManager

# Instrument FastAPI with OpenTelemetry
try:
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    has_otel = True
except ImportError:
    has_otel = False

app = FastAPI(
    title="deepBAG Enterprise Gateway & Semantic Tool Router",
    version="2.0.0",
    description="Production API Gateway translating natural language into validated NBA API queries with sliding-window rate limiting, distributed caching, and OTel/Langfuse telemetry."
)

if has_otel:
    FastAPIInstrumentor.instrument_app(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or os.environ.get("NEBIUS_API_KEY")
ACTIVE_PORT = int(os.environ.get("PORT", "8000"))

print("Initializing NBA Query Analyzer in FastAPI app...")
try:
    if not API_KEY:
        print("Notice: No LLM API key environment variable is set. Running in offline mode.")
        analyzer = None
    else:
        analyzer = NBAQueryAnalyzer(api_key=API_KEY)
        print("Analyzer initialization complete.")
except Exception as e:
    print(f"Failed to initialize analyzer: {e}")
    analyzer = None


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    if not analyzer:
        raise HTTPException(status_code=500, detail="Analyzer failed to initialize. Check API key and dependencies.")
        
    async def event_generator():
        with TelemetryManager.start_span("deepbag.chat_stream", {"query": request.query, "session_id": request.session_id or ""}):
            try:
                print(f"[INGRESS VALIDATED] Received query (model={request.model}): {request.query}")
                for event in analyzer.process_query_agentic(request.query, model_name=request.model):
                    yield f"data: {json.dumps(event)}\n\n"

            except Exception as e:
                import traceback
                traceback.print_exc()
                yield f"data: {json.dumps({'type': 'final_answer', 'content': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/api/status")
async def status_endpoint():
    cache_stats = analyzer.cache.get_stats() if analyzer and hasattr(analyzer, "cache") else {}
    return {
        "status": "online",
        "gateway_version": "2.0.0",
        "analyzer_initialized": analyzer is not None,
        "cache": cache_stats,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
