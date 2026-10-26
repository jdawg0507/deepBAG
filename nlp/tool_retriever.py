import numpy as np
import google.generativeai as genai

import re
from collections import Counter
import math

class EndpointToolRetriever:
    def __init__(self, endpoint_library, model_name=None):
        self.endpoint_library = endpoint_library
        self.endpoint_names = []
        self.documents = []
        self.idf = {}
        self._initialize_search_index()

    def _initialize_search_index(self):
        """Precompute local TF-IDF style index for lightning fast, free keyword search"""
        print(f"Building local keyword index for {len(self.endpoint_library)} endpoints...")
        
        self.endpoint_names = []
        tokenized_docs = []
        
        for name, info in self.endpoint_library.items():
            self.endpoint_names.append(name)
            params_list = list(info.get('parameters', {}).keys())
            desc = f"{name} {info.get('description', '')} {' '.join(params_list)}"
            
            # Simple tokenization
            tokens = re.findall(r'\w+', desc.lower())
            tokenized_docs.append(tokens)
            self.documents.append(tokens)
            
        # Compute IDF
        num_docs = len(tokenized_docs)
        doc_frequencies = Counter()
        for tokens in tokenized_docs:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                doc_frequencies[token] += 1
                
        for token, freq in doc_frequencies.items():
            self.idf[token] = math.log(num_docs / (1 + freq))
            
        print("Successfully built local keyword index in memory (0 API calls!).")

    def retrieve_tools(self, query, top_k=5):
        """Retrieve the top_k most relevant endpoints using pure local keyword search"""
        query_tokens = re.findall(r'\w+', query.lower())
        
        scores = []
        for i, doc_tokens in enumerate(self.documents):
            doc_counts = Counter(doc_tokens)
            score = 0
            for q_token in query_tokens:
                if q_token in doc_counts:
                    # TF-IDF calculation
                    tf = doc_counts[q_token]
                    idf = self.idf.get(q_token, 0)
                    score += tf * idf
            scores.append((score, self.endpoint_names[i]))
            
        # Sort by score descending
        scores.sort(key=lambda x: x[0], reverse=True)
        
        top_endpoints = [name for score, name in scores[:top_k] if score > 0]
        
        # Fallback if no keywords matched
        if not top_endpoints:
            return ["PlayerGameLogs", "LeagueDashPlayerStats"][:top_k]
            
        return top_endpoints
