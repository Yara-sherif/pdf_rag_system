SYSTEM_PROMPT = """You are an expert on the textbook "Introduction to Natural Language Processing".

Answer questions using ONLY the context passages provided to you.
Give a thorough, detailed answer that fully explains the concept.
Use examples from the context where available.
Structure your answer clearly — definition first, then elaboration, then examples.
If the answer is not in the context, or the question is unrelated to the document, respond with exactly:
"Sorry this question is not related with the document content."

Be accurate and factual."""

REWRITE_PROMPT = """You are a query rewriting assistant for a study companion RAG system.

The knowledge base is an English-language NLP textbook. All documents are in English.

Rewrite the user's question so that:
1. It is in English — if the query is in another language, translate it to English,
   because all retrieval is against English passages and a non-English query will
   degrade embedding similarity scores.
2. It is self-contained — resolve pronouns and references to prior conversation
   (e.g. "it", "that algorithm", "the previous one") using the conversation history.
3. It is precise and clean — remove filler words, fix typos, make the intent explicit.

Return ONLY the rewritten query. No explanation, no preamble, no quotes.
If the query is already clear, standalone, and in English, return it unchanged."""


def query_rewrite_extend(user_input: str, chat_history: list) -> str:
    history_str = ""
    if chat_history:
        for i, msg in enumerate(chat_history[-4:]):
            role = "User" if i % 2 == 0 else "Assistant"
            content = msg.content if hasattr(msg, "content") else str(msg)
            history_str += f"{role}: {content}\n"

    return f"""Conversation history (for reference only):
{history_str or "No previous history."}

Original user query: {user_input}

Rewritten query:"""


def system_prompt_extend(user_input: str, chat_history: list, context_docs: list) -> str:
    history_str = ""
    if chat_history:
        recent = chat_history[-6:]
        for i, msg in enumerate(recent):
            role = "User" if i % 2 == 0 else "Assistant"
            content = msg.content if hasattr(msg, "content") else str(msg)
            history_str += f"{role}: {content}\n"

    context_str = "\n\n".join(
        f"[Page {doc['page_number']}]:\n{doc['text']}"
        for doc in context_docs
    ) if context_docs else "No context available."

    return f"""Conversation history:
{history_str or "No previous history."}

Context from the NLP textbook:
{context_str}

Question: {user_input}

Answer:"""
