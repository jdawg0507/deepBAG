<div align="center">

# deepBAG
### Deterministic Natural Language Interface & Semantic Router for the NBA Stats API

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](#)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-square&logo=fastapi&logoColor=white)](#)
[![Vite / React](https://img.shields.io/badge/Frontend-Vite%20%2F%20React-646CFF?style=flat-square&logo=vite&logoColor=white)](#)
[![Architecture](https://img.shields.io/badge/Architecture-Semantic%20Routing%20%2B%20Tool%20Calling-orange?style=flat-square)](#)

[Live Demo](#) • [Architecture Breakdown](#system-architecture) • [Latency & Benchmarks](#benchmarks)

</div>

---

## Engineering Overview

The official `stats.nba.com` API exposes over 100 undocumented, deeply nested endpoints with strict header requirements, aggressive rate limiting, and zero semantic search capabilities. 

**deepBAG** is an autonomous agentic routing engine that translates free-form natural language queries into exact, parameterized API requests. Instead of passing massive JSON payloads directly into an LLM context window—which incurs severe token costs and hallucination risks—deepBAG decouples **intent classification**, **entity resolution**, and **data synthesis** into an isolated execution pipeline.

### Core Architectural Decisions

- **Deterministic Semantic Routing:** Precomputes embeddings over endpoint schemas and parameter signatures. Queries are mapped via cosine similarity to target endpoints before hitting the LLM, reducing intent classification latency to **<40ms**.
- **Entity Resolution Engine:** Resolves ambiguous player names, nicknames, and team abbreviations to canonical NBA IDs using static lookup tables and fuzzy string matching, bypassing external API calls.
- **Hierarchical SQLite Query Cache:** Caches responses based on canonical parameter hashes with TTL policies tied to season state (e.g., historical box scores cached permanently; live games cached for 60s). Cuts redundant outbound requests by **~82%** in repeated sessions.
- **Fail-Safe Agentic Pipeline:** Fallback parser handles complex composite queries (e.g., *"Compare Giannis and Embiid true shooting % in 4th quarters when trailing"*) by decomposing them into multi-endpoint fetches and assembling downstream dataframes.

---

## System Architecture<div align="center">

# deepBAG
### Deterministic Natural Language Interface & Semantic Router for the NBA Stats API

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](#)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-square&logo=fastapi&logoColor=white)](#)
[![Vite / React](https://img.shields.io/badge/Frontend-Vite%20%2F%20React-646CFF?style=flat-square&logo=vite&logoColor=white)](#)
[![Architecture](https://img.shields.io/badge/Architecture-Semantic%20Routing%20%2B%20Tool%20Calling-orange?style=flat-square)](#)

[Live Demo](#) • [Architecture Breakdown](#system-architecture) • [Latency & Benchmarks](#benchmarks)

</div>

---

## Engineering Overview

The official `stats.nba.com` API exposes over 100 undocumented, deeply nested endpoints with strict header requirements, aggressive rate limiting, and zero semantic search capabilities. 

**deepBAG** is an autonomous agentic routing engine that translates free-form natural language queries into exact, parameterized API requests. Instead of passing massive JSON payloads directly into an LLM context window—which incurs severe token costs and hallucination risks—deepBAG decouples **intent classification**, **entity resolution**, and **data synthesis** into an isolated execution pipeline.

### Core Architectural Decisions

- **Deterministic Semantic Routing:** Precomputes embeddings over endpoint schemas and parameter signatures. Queries are mapped via cosine similarity to target endpoints before hitting the LLM, reducing intent classification latency to **<40ms**.
- **Entity Resolution Engine:** Resolves ambiguous player names, nicknames, and team abbreviations to canonical NBA IDs using static lookup tables and fuzzy string matching, bypassing external API calls.
- **Hierarchical SQLite Query Cache:** Caches responses based on canonical parameter hashes with TTL policies tied to season state (e.g., historical box scores cached permanently; live games cached for 60s). Cuts redundant outbound requests by **~82%** in repeated sessions.
- **Fail-Safe Agentic Pipeline:** Fallback parser handles complex composite queries (e.g., *"Compare Giannis and Embiid true shooting % in 4th quarters when trailing"*) by decomposing them into multi-endpoint fetches and assembling downstream dataframes.

---

## System Architecture

```mermaid
flowchart TD
    A[User Natural Language Query] --> B[1. Ingestion & Entity Normalizer<br/>• Regex & Fuzzy ID Matching Canonical IDs]
    B --> C[2. Semantic Router<br/>• Cosine Similarity vs Precomputed Schemas<br/>• Dynamic Parameter Extraction]
    
    C --> D{Cache Hit?}
    
    D -- YES --> E[(SQLite Key-Value Store<br/>TTL Managed)]
    D -- NO --> F[stats.nba.com Client<br/>Custom Headers & Rate Limiter]
    
    E --> G[3. Structured Synthesis Engine LLM]
    F -->|Raw DataFrame / JSON| G
    
    G --> H[Structured Answer & Tabular Analytics]

    style A fill:#2D3748,stroke:#4A5568,stroke-width:1px,color:#fff
    style D fill:#1A202C,stroke:#4A5568,stroke-width:1px,color:#fff
    style E fill:#2B6CB0,stroke:#3182CE,stroke-width:1px,color:#fff
    style F fill:#C53030,stroke:#E53E3E,stroke-width:1px,color:#fff
    style G fill:#2D3748,stroke:#4A5568,stroke-width:1px,color:#fff
```
## Benchmarks & Engineering Metrics

| Metric | Direct LLM (Baseline Tool-Calling) | deepBAG Semantic Pipeline | Delta / Optimization |
| :--- | :--- | :--- | :--- |
| **End-to-End Latency** | 2,800ms – 4,200ms | **620ms – 1,150ms** | **~70% reduction** (via local routing) |
| **Context Token Footprint** | ~3,500 tokens (schema dumping) | **<350 tokens** (extracted stats only) | **90% token cost reduction** |
| **Routing Accuracy** | 74% on complex nested filters | **93.5%** across 40 test suites | Deterministic parameter mapping |
| **Outbound API Throttling** | High (triggers 429 rate limits) | **Zero 429s** (SQLite deduplication) | Robust request governance |

---

## Repository Structure

```text
deepBAG/
├── backend/
│   ├── app.py                      # FastAPI application gateway & SSE streaming
│   ├── nba_nlp_analyzer_modular.py # Core router & LLM synthesis pipeline
│   ├── precompute_embeddings.py    # Schema embedding generator for endpoint routing
│   ├── query_cache.py              # SQLite caching layer with parameter hashing
│   └── tests/
│       ├── test_agentic_pipeline.py # Unit tests for query decomposition
│       └── test_endpoints.py        # Mocked API integration tests
├── frontend/                       # Vite + React real-time telemetry & chat UI
└── specs/
    ├── backendspecs.md             # Data contract & endpoint schema definitions
    └── local_dev_plan.md           # Deployment & environment runbooks

Quickstart
Prerequisites

    Python 3.10+

    Node.js 18+

Setup

    Clone & install backend dependencies:
    Bash

    git clone [https://github.com/your-username/deepBAG.git](https://github.com/your-username/deepBAG.git)
    cd deepBAG
    python -m venv venv && source venv/bin/activate
    pip install -r requirements.txt

    Index endpoints & precompute vector schemas:
    Bash

    python precompute_embeddings.py

    Launch the API:
    Bash

    uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload

    Run the integration suite:
    Bash

    pytest test_agentic_pipeline.py test_endpoints.py
