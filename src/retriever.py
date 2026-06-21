"""
Hybrid retriever: FAISS (dense semantic) + TF-IDF (sparse lexical).
Results merged via Reciprocal Rank Fusion (RRF).
"""
import json
import pickle
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

_ROOT = Path(__file__).parent.parent
PAGES_PATH = _ROOT / "data" / "pages.json"
FAISS_PATH = _ROOT / "indexes" / "faiss_index.bin"
TFIDF_VEC_PATH = _ROOT / "indexes" / "tfidf_vectorizer.pkl"
TFIDF_MAT_PATH = _ROOT / "indexes" / "tfidf_matrix.npy"
# BGE base is fine-tuned for retrieval (query→passage), not just sentence similarity.
# Dimension: 768. Must match the model used in ingest.py.
MODEL_NAME = "BAAI/bge-base-en-v1.5"

# BGE instruction prefix: applied to queries only, NOT to stored document embeddings.
# This asymmetry is intentional — BGE was trained with this distinction and retrieval
# F1 improves measurably when the prefix is used only on the query side.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# RRF constant — lower values give more weight to highly-ranked results.
# Standard value is 60 (Cormack et al. 2009), but for short queries
# against long documents, 10 better amplifies the semantic FAISS ranking
# and prevents low-ranked keyword matches from outweighing strong dense hits.
RRF_K = 10


class HybridRetriever:
    def __init__(self):
        self.index = None
        self.tfidf = None
        self.tfidf_matrix = None
        self.pages = None
        self.encoder = None

    def load_index(self) -> None:
        self.encoder = SentenceTransformer(MODEL_NAME)
        self.index = faiss.read_index(str(FAISS_PATH))
        with open(TFIDF_VEC_PATH, "rb") as f:
            self.tfidf = pickle.load(f)
        self.tfidf_matrix = np.load(TFIDF_MAT_PATH)
        with open(PAGES_PATH, encoding="utf-8") as f:
            self.pages = json.load(f)

    def faiss_search(self, query: str, k: int = 10) -> list[tuple[float, int]]:
        """Dense semantic search — returns (score, page_index) pairs."""
        q_vec = self.encoder.encode(
            [BGE_QUERY_PREFIX + query], normalize_embeddings=True
        ).astype(np.float32)
        scores, indices = self.index.search(q_vec, k)
        return [(float(scores[0][i]), int(indices[0][i])) for i in range(len(indices[0]))]

    def tfidf_search(self, query: str, k: int = 10) -> list[tuple[float, int]]:
        """Sparse lexical search — returns (score, page_index) pairs."""
        q_vec = self.tfidf.transform([query])
        sims = sklearn_cosine(q_vec, self.tfidf_matrix)[0]
        top_k = np.argsort(sims)[::-1][:k]
        return [(float(sims[i]), int(i)) for i in top_k]

    def _rrf_merge(self, *rankings: list[tuple[float, int]]) -> list[tuple[int, float]]:
        """
        Reciprocal Rank Fusion across multiple ranked lists.
        score(d) = Σ_i  1 / (rank_i(d) + RRF_K)
        Returns list of (page_index, rrf_score) sorted descending.
        """
        scores: dict[int, float] = {}
        for ranked_list in rankings:
            for rank, (_, idx) in enumerate(ranked_list):
                scores[idx] = scores.get(idx, 0.0) + 1.0 / (rank + 1 + RRF_K)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def hybrid_search(self, query: str, k: int = 5) -> dict:
        """
        Retrieve the top-k most relevant pages using hybrid FAISS + TF-IDF search.
        Returns a dict with three keys:
          - "faiss": top-k dense results with faiss_score
          - "tfidf": top-k sparse results with tfidf_score
          - "final": top-k RRF-fused results with rrf_score
        """
        candidate_k = min(k * 3, len(self.pages))
        faiss_res = self.faiss_search(query, candidate_k)
        tfidf_res = self.tfidf_search(query, candidate_k)

        faiss_docs = []
        for score, idx in faiss_res[:k]:
            doc = dict(self.pages[idx])
            doc["faiss_score"] = round(score, 4)
            faiss_docs.append(doc)

        tfidf_docs = []
        for score, idx in tfidf_res[:k]:
            doc = dict(self.pages[idx])
            doc["tfidf_score"] = round(score, 4)
            tfidf_docs.append(doc)

        fused = self._rrf_merge(faiss_res, tfidf_res)
        final_docs = []
        for idx, rrf_score in fused[:k]:
            doc = dict(self.pages[idx])
            doc["rrf_score"] = round(rrf_score, 4)
            final_docs.append(doc)

        return {"faiss": faiss_docs, "tfidf": tfidf_docs, "final": final_docs}
