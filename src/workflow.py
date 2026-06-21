from agents import retriever_agent, response_agent, rewrite_query_agent
from models import State
from langgraph.graph import StateGraph, START, END


class Workflow:
    def __init__(self):
        self.rewrite_query_agent = rewrite_query_agent
        self.retriever_agent = retriever_agent
        self.response_agent = response_agent

    def _build_graph(self):
        graph = StateGraph(State)

        graph.add_node("rewrite_query", self.rewrite_query_agent)
        graph.add_node("retriever_agent", self.retriever_agent)
        graph.add_node("response_agent", self.response_agent)

        graph.add_edge(START, "rewrite_query")
        graph.add_edge("rewrite_query", "retriever_agent")
        graph.add_edge("retriever_agent", "response_agent")
        graph.add_edge("response_agent", END)

        return graph.compile()

    def run(self, initial_state: State) -> dict:
        graph = self._build_graph()
        result = graph.invoke(initial_state)
        return result
