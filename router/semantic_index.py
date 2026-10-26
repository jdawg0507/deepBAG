"""
Dense Candidate Retrieval Index using FastEmbed ONNX embeddings.
Embeds all 135 endpoint capability_cards into data/endpoint_embeddings.npz.
Provides fast cosine-similarity Top-K candidate endpoint retrieval at runtime.
"""
import os
import json
import logging
import numpy as np
from typing import List, Dict, Any, Tuple
from fastembed import TextEmbedding

from catalogue.schema import EndpointSpec

logger = logging.getLogger("deepbag.semantic_index")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CATALOGUE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "endpoint_catalogue_v2.json")
INDEX_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "endpoint_embeddings.npz")

class SemanticEndpointIndex:
    def __init__(
        self,
        catalogue_path: str = CATALOGUE_PATH,
        index_path: str = INDEX_PATH,
        model_name: str = EMBEDDING_MODEL,
    ):
        self.catalogue_path = catalogue_path
        self.index_path = index_path
        self.model = TextEmbedding(model_name=model_name)
        self.catalogue: Dict[str, Dict[str, Any]] = self._load_catalogue()
        self.endpoint_names: List[str] = []
        self.embeddings: np.ndarray = np.empty((0, 384))
        self._load_or_build_index()

    def _load_catalogue(self) -> Dict[str, Dict[str, Any]]:
        if os.path.exists(self.catalogue_path):
            with open(self.catalogue_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _load_or_build_index(self):
        if os.path.exists(self.index_path):
            data = np.load(self.index_path)
            self.endpoint_names = list(data["endpoint_names"])
            self.embeddings = data["embeddings"]
            logger.info(f"Loaded {len(self.endpoint_names)} endpoint vectors from {self.index_path}")
        else:
            self.build_index()

    def build_index(self):
        """Generates dense embeddings for all endpoint capability cards and saves to .npz"""
        logger.info(f"Building semantic vector index for {len(self.catalogue)} endpoints...")
        names = []
        cards = []
        for ep_name, spec in self.catalogue.items():
            names.append(ep_name)
            cards.append(spec.get("capability_card", f"Endpoint {ep_name}"))

        if not cards:
            return

        embs_generator = self.model.embed(cards)
        vectors = np.array(list(embs_generator), dtype=np.float32)
        # Normalize vectors for fast inner-product cosine similarity
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized_vectors = vectors / norms

        self.endpoint_names = names
        self.embeddings = normalized_vectors

        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        np.savez_compressed(
            self.index_path,
            endpoint_names=np.array(self.endpoint_names),
            embeddings=self.embeddings
        )
        logger.info(f"Successfully saved endpoint vectors to {self.index_path}")

    def retrieve_candidates(
        self,
        query: str,
        top_k: int = 5,
        shadow_keyword_matcher: Any = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Embeds the query and retrieves the Top-K candidate endpoints by cosine similarity.
        Runs shadow mode on keyword matcher for telemetry.
        """
        if self.embeddings.shape[0] == 0:
            return []

        # 1. Embed query
        q_vec = np.array(list(self.model.embed([query])), dtype=np.float32)[0]
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        # 2. Cosine similarity via dot product
        similarities = np.dot(self.embeddings, q_vec)
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            name = self.endpoint_names[idx]
            score = float(similarities[idx])
            spec = self.catalogue.get(name, {})
            results.append((name, score, spec))

        # 3. Shadow mode telemetry
        if shadow_keyword_matcher:
            try:
                shadow_plan = shadow_keyword_matcher(query)
                semantic_top5_names = [r[0] for r in results]
                keyword_hit = shadow_plan.endpoint if hasattr(shadow_plan, "endpoint") else None
                logger.info(
                    "keyword_fallback_delta",
                    extra={
                        "semantic_top5": semantic_top5_names,
                        "keyword_hit": keyword_hit,
                        "in_top5": keyword_hit in semantic_top5_names if keyword_hit else None
                    }
                )
            except Exception:
                pass
        return results

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        candidates = self.retrieve_candidates(query=query, top_k=top_k)
        return [{"endpoint_name": c[0], "score": c[1], "spec": c[2]} for c in candidates]

if __name__ == "__main__":
    idx = SemanticEndpointIndex()
    idx.build_index()
    sample_q = "Compare LeBron and Curry scoring over their last 5 games"
    candidates = idx.retrieve_candidates(sample_q, top_k=5)
    print(f"\nTop candidates for '{sample_q}':")
    for name, score, _ in candidates:
        print(f"  {score:.4f} -> {name}")
