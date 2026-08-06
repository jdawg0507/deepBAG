# NBA API LLM Interface - Backend Specifications (Embedding-Based Design)

## Core Architecture
1. **Embedding Engine**: Gemini Embeddings (precomputed parameter/endpoint mappings)
2. **LLM Context Handler**: Gemini Pro (for nickname resolution, colloquialisms, and complex queries)
3. **API Integration**: `nba_api` library with embedding-optimized routing
4. **Processing Flow**: Query → Parameter Extraction → Endpoint Matching → API Call → Formatted Response

---

## Key Components

### 1. Parameter Embedding System
**Responsibilities**:
- Precompute embeddings for all parameters from `parameters.py`
- Store embeddings in optimized format (vector DB or serialized)
- Handle parameter similarity matching

**Optimizations**:
- Hierarchical embeddings for parameter categories
- Aliases and synonyms included in precomputed embeddings
- Version-controlled embeddings for parameter updates

### 2. Endpoint Embedding System
**Responsibilities**:
- Precompute embeddings for all API endpoints
- Map endpoint capabilities to parameter requirements
- Handle endpoint similarity matching

**Optimizations**:
- Endpoint descriptions include supported parameters
- Embeddings capture parameter-endpoint relationships
- Version-controlled for API changes

### 3. Contextual LLM Layer
**Responsibilities**:
- Resolve player nicknames ("King James" → "LeBron James")
- Handle colloquialisms ("dimes" → "assists")
- Process complex queries with multiple intents
- Provide query suggestions for ambiguous inputs

**Optimizations**:
- Maintain context across follow-up queries
- Use precomputed embeddings for faster resolution
- Cache common nickname mappings

---

## Processing Pipeline

1. **Input Processing**:
   - User query received
   - LLM handles initial context (nicknames, colloquialisms)

2. **Parameter Extraction**:
   - Query embedding compared against precomputed parameter embeddings
   - Top matching parameters selected

3. **Endpoint Matching**:
   - Extracted parameters compared against endpoint embeddings
   - Most suitable endpoint selected

4. **API Execution**:
   - Parameters formatted for selected endpoint
   - API call executed
   - Response received

5. **Response Formatting**:
   - Raw API response processed
   - Contextual LLM adds explanations/insights
   - Final response formatted for user

---

## Optimization Strategy

1. **Precomputation**:
   - Parameter and endpoint embeddings generated at build time
   - Stored in optimized format for fast access

2. **Caching**:
   - Common query patterns cached
   - Frequent API responses cached

3. **Hybrid Processing**:
   - Embeddings handle direct parameter/endpoint matching
   - LLM handles complex context and edge cases

---

## Initial Endpoints

1. **POST /query**
   - Accepts natural language queries
   - Returns formatted responses
   - Uses embedding-based processing pipeline

2. **GET /status**
   - Service health check
   - Embedding system status
   - LLM model status

---

## Development Phases

### Phase 1: Core Embedding System
- Implement parameter embedding precomputation
- Build endpoint embedding system
- Basic parameter extraction

### Phase 2: Contextual LLM Integration
- Add nickname resolution
- Handle colloquialisms
- Process complex queries

### Phase 3: Optimization
- Implement caching layer
- Add query suggestions
- Improve error handling

---

## Key Decisions
1. **Embedding Model**: Use Gemini's embedding-001 for consistency
2. **Precomputation**: Generate embeddings at build time for performance
3. **Hybrid Approach**: Combine embeddings for efficiency with LLM for context
4. **Version Control**: Track parameter/endpoint changes to regenerate embeddings

---

This specification provides a foundation for your embedding-based backend while maintaining the abstraction you want. Would you like to adjust any part of this before we proceed to implementation?