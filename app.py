import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add the backend directory to sys.path so we can import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from nba_nlp_analyzer_modular import NBAQueryAnalyzer

app = FastAPI(title="NBA API Agent Chat")

# Add CORS middleware to allow requests from the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the exact frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the analyzer globally so it loads the API and embeddings once
# Using the fallback key from main.py if environment variable is not set
API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyDGvmWXOZRTyqWl_MhcYEAU2jvKV5mS8vA")

print("Initializing NBA Query Analyzer in FastAPI app...")
try:
    analyzer = NBAQueryAnalyzer(api_key=API_KEY)
    print("Initialization complete.")
except Exception as e:
    print(f"Failed to initialize analyzer: {e}")
    analyzer = None

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    response: str

from fastapi.responses import StreamingResponse
import json

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    if not analyzer:
        raise HTTPException(status_code=500, detail="Analyzer failed to initialize. Check API key and dependencies.")
        
    async def event_generator():
        try:
            print(f"Received query: {request.query}")
            for event in analyzer.process_query_agentic(request.query):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'final_answer', 'content': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/api/status")
async def status_endpoint():
    return {"status": "online", "analyzer_initialized": analyzer is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
