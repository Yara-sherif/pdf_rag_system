from typing import Optional, Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class State(TypedDict):
    query: str
    rewritten_query: str
    chat_history: Annotated[list, add_messages]
    faiss_context: Optional[list]
    tfidf_context: Optional[list]
    context: Optional[list]
    response: str
