from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from models import State
from prompts import SYSTEM_PROMPT, REWRITE_PROMPT, system_prompt_extend, query_rewrite_extend
from retriever import HybridRetriever
from dotenv import load_dotenv

load_dotenv()

# Low temperature → deterministic, factual answers.
# High top_p → retains technical vocabulary while filtering improbable tokens.
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.1,
    top_p=0.95,
)

# Load retrieval indexes once at module import (avoids reloading per request).
_retriever = HybridRetriever()
_retriever.load_index()


def rewrite_query_agent(state: State) -> dict:
    user_input = state.get("query")
    chat_history = state.get("chat_history", [])

    messages = [
        SystemMessage(content=REWRITE_PROMPT),
        HumanMessage(content=query_rewrite_extend(user_input, chat_history)),
    ]

    response = llm.invoke(messages)
    return {"rewritten_query": response.content.strip()}


def retriever_agent(state: State) -> dict:
    # Use the rewritten query for retrieval; fall back to the original if unavailable.
    query = state.get("rewritten_query") or state.get("query")
    results = _retriever.hybrid_search(query, k=7)
    return {
        "faiss_context": results["faiss"],
        "tfidf_context": results["tfidf"],
        "context": results["final"],
    }


def response_agent(state: State) -> dict:
    # Answer using the rewritten (cleaner, possibly translated) query.
    user_input = state.get("rewritten_query") or state.get("query")
    chat_history = state.get("chat_history", [])
    context = state.get("context", [])

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=system_prompt_extend(user_input, chat_history, context)),
    ]

    response = llm.invoke(messages)
    return {"response": response.content}
