import os
import sys

# Ensure correct path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api.endpoint_utils import build_endpoint_library
from api.endpoint_descriptions import enhance_endpoint_descriptions
from nlp.tool_retriever import EndpointToolRetriever
import google.generativeai as genai

# Configure API Key securely from environment variable (optional for local TF-IDF search)
api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

print("1. Building NBA API endpoint library...")
lib = build_endpoint_library()
lib = enhance_endpoint_descriptions(lib)

print(f"2. Loaded {len(lib)} endpoints. Starting index precomputation...")
retriever = EndpointToolRetriever(lib)
print("3. Precomputation complete!\n")

# Demonstration: Test semantic tool retrieval across sample queries
sample_queries = [
    "player shot tracking and shot chart",
    "defensive ratings and rim protection",
    "recent game logs and box scores",
    "team standings and win streaks",
]

print("=== Demonstration: Semantic Endpoint Retrieval ===")
for q in sample_queries:
    top_tools = retriever.retrieve_tools(q, top_k=3)
    print(f"\nQuery: '{q}'")
    print(f"  -> Recommended Endpoints: {top_tools}")
