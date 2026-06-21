import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
from workflow import Workflow
from models import State
from langchain_core.messages import HumanMessage

st.set_page_config(
    page_title="NLP Textbook Q&A",
    page_icon="📚",
    layout="wide",
)

st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("📚 Introduction to NLP — Q&A Chatbot")
st.caption("Powered by Gemini 2.5 Flash · FAISS + TF-IDF hybrid retrieval · Query rewriting · Page-level citations")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("About")
    st.markdown("""
    This chatbot answers questions grounded in the textbook
    **"Introduction to Natural Language Processing"**.

    **Pipeline:**
    1. Query rewriting: Gemini cleans and translates the query to English
    2. Hybrid retrieval: FAISS (dense) + TF-IDF (sparse) via RRF
    3. Answer generation: Gemini 2.5 Flash (temp=0.1, top_p=0.95)

    **Sample questions:**
    - What is tokenization?
    - What corpora types exist?
    - How many synsets does WordNet have?
    - Difference between stemming and lemmatization?
    - What is a context-free grammar?
    """)
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()


def render_sources(faiss_docs, tfidf_docs, final_docs):
    with st.expander("🔍 FAISS Sources (Dense Retrieval)"):
        for doc in faiss_docs:
            st.markdown(f"**Page {doc['page_number']}** *(score: {doc['faiss_score']})*")
            st.caption(doc["text"][:350] + "…")
    with st.expander("📝 TF-IDF Sources (Lexical Retrieval)"):
        for doc in tfidf_docs:
            st.markdown(f"**Page {doc['page_number']}** *(score: {doc['tfidf_score']})*")
            st.caption(doc["text"][:350] + "…")
    with st.expander("⚡ Final Sources (RRF Fused)"):
        for doc in final_docs:
            st.markdown(f"**Page {doc['page_number']}** *(relevance score: {doc['rrf_score']})*")
            st.caption(doc["text"][:350] + "…")


# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("rewritten_query"):
            st.caption(f"Query rewritten to: *\"{msg['rewritten_query']}\"*")
        if msg.get("sources"):
            render_sources(
                msg.get("faiss_sources", []),
                msg.get("tfidf_sources", []),
                msg["sources"],
            )

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about NLP concepts, algorithms, corpora…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Rewriting query, retrieving context, and generating answer…"):
            chat_history = [
                HumanMessage(content=m["content"])
                for m in st.session_state.messages
                if m["role"] == "user"
            ]
            initial_state = State(
                chat_history=chat_history,
                query=prompt,
                rewritten_query="",
                faiss_context=None,
                tfidf_context=None,
                context=None,
                response="",
            )
            result = Workflow().run(initial_state)
            response = result.get("response", "")
            rewritten_query = result.get("rewritten_query", "")
            faiss_context = result.get("faiss_context", [])
            tfidf_context = result.get("tfidf_context", [])
            context = result.get("context", [])

        st.markdown(response)

        # Show rewrite info when the query was meaningfully changed.
        if rewritten_query and rewritten_query.strip().lower() != prompt.strip().lower():
            st.caption(f"Query rewritten to: *\"{rewritten_query}\"*")

        if context:
            render_sources(faiss_context, tfidf_context, context)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "rewritten_query": rewritten_query if rewritten_query.strip().lower() != prompt.strip().lower() else "",
        "faiss_sources": faiss_context,
        "tfidf_sources": tfidf_context,
        "sources": context,
    })
