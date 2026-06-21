# RAG Pipeline — Introduction to NLP

A Retrieval-Augmented Generation system for Q&A over an NLP textbook.

## Stack

| Component | Choice |
|---|---|
| LLM | Gemini 2.5 Flash (temperature=0.1, top_p=0.95) |
| Query rewriting | Gemini 2.5 Flash (translates + cleans query before retrieval) |
| Embeddings | BAAI/bge-base-en-v1.5 (768-dim, retrieval-tuned) |
| Query encoding | BGE instruction prefix (asymmetric query/document encoding) |
| Dense index | FAISS IndexFlatIP |
| Sparse index | TF-IDF (bigrams, sublinear TF) |
| Merge strategy | Reciprocal Rank Fusion (RRF, K=10) |
| Framework | LangGraph (rewrite → retrieve → generate) |
| UI | Streamlit |

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add your Gemini API key
cp .env.example .env
# Edit .env: GOOGLE_API_KEY=your_key_here

# 3. Build indexes from the PDF (run once; re-run if model or data changes)
python pipeline/preprocessing.py   # extracts text → data/pages.json
python pipeline/ingest.py          # embeds with BGE, builds indexes → indexes/

# 4. Run the Streamlit UI
streamlit run app.py

```

## Graph Architecture

```
START → rewrite_query → retriever_agent → response_agent → END
```