"""
Integration tests for FastAPI Gateway endpoints (/api/status, /api/chat).
"""
import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_gateway_status_endpoint():
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["gateway_version"] == "2.0.0"
    assert "cache" in data

def test_gateway_chat_empty_query_rejected():
    # Strict Pydantic validation rejects empty query with 422 Unprocessable Entity
    response = client.post("/api/chat", json={"query": " "})
    assert response.status_code == 422

def test_gateway_chat_extra_field_rejected():
    # Strict Pydantic validation forbids extra fields
    response = client.post("/api/chat", json={"query": "Valid query", "invalid_extra": 123})
    assert response.status_code == 422
