import os
import sys

# Ensure correct path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api.endpoint_utils import build_endpoint_library
from api.endpoint_descriptions import enhance_endpoint_descriptions
from nlp.tool_retriever import EndpointToolRetriever
import google.generativeai as genai

# Configure API Key
api_key = os.environ.get("GOOGLE_API_KEY", "AIzaSyAHrY15BOlnq6d5fEMEOoYauwrQGOUJfX8")
genai.configure(api_key=api_key)

print("Building endpoint library...")
lib = build_endpoint_library()
lib = enhance_endpoint_descriptions(lib)

print(f"Loaded {len(lib)} endpoints. Starting precomputation...")
retriever = EndpointToolRetriever(lib)
print("Done!")
