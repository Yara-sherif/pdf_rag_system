"""
Stage 2 — Index: embed page chunks and build FAISS + TF-IDF indexes.
Run once after preprocessing.py: python ingest.py
Outputs: faiss_index.bin, tfidf_vectorizer.pkl, tfidf_matrix.npy
"""
import json
import pickle
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

_ROOT = Path(__file__).parent.parent
PAGES_PATH = _ROOT / "data" / "pages.json"
FAISS_PATH = _ROOT / "indexes" / "faiss_index.bin"
TFIDF_VEC_PATH = _ROOT / "indexes" / "tfidf_vectorizer.pkl"
TFIDF_MAT_PATH = _ROOT / "indexes" / "tfidf_matrix.npy"
# BGE base is fine-tuned on query-passage pairs (MS-MARCO, NLI), making it
# significantly more accurate for retrieval than general sentence-similarity models.
# Dimension: 768 (vs 384 for MiniLM).
MODEL_NAME = "BAAI/bge-base-en-v1.5"


def build_indexes(pages: list[dict]) -> None:
    texts = [p["text"] for p in pages]

    print(f"Embedding {len(texts)} pages with {MODEL_NAME}...")
    encoder = SentenceTransformer(MODEL_NAME)
    embeddings = encoder.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = embeddings.astype(np.float32)

    print("Building FAISS IndexFlatIP (cosine similarity on unit vectors)...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    FAISS_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_PATH))
    print(f"  FAISS index saved → {FAISS_PATH}  ({index.ntotal} vectors, dim={dim})")

    print("Building TF-IDF index (bigrams, sublinear TF)...")
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)
    tfidf_matrix = tfidf.fit_transform(texts)
    with open(TFIDF_VEC_PATH, "wb") as f:
        pickle.dump(tfidf, f)
    np.save(TFIDF_MAT_PATH, tfidf_matrix.toarray())
    print(f"  TF-IDF saved → {TFIDF_VEC_PATH}, {TFIDF_MAT_PATH}")


if __name__ == "__main__":
    with open(PAGES_PATH, encoding="utf-8") as f:
        pages = json.load(f)
    print(f"Loaded {len(pages)} pages from {PAGES_PATH}")
    build_indexes(pages)
    print("\nIngestion complete. Ready for retrieval.")
