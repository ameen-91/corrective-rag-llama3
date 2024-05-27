from typing_extensions import TypedDict
from typing import List
from langgraph.graph import StateGraph, END
from .nodes import (
    retrieve,
    grade_documents,
    transform_query,
    web_search_node,
    generate,
    decide_to_generate,
)


class GraphState(TypedDict):

    question: str
    generation: str
    web_search: str
    documents: List[str]


def corrective_rag():
    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("web_search_node", web_search_node)
    workflow.add_node("generate", generate)

    workflow.set_entry_point("transform_query")
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {"generate": "generate", "web_search_node": "web_search_node"},
    )
    workflow.add_edge("web_search_node", "generate")
    workflow.add_edge("generate", END)

    app = workflow.compile()

    return app


def simple_rag():

    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    app = workflow.compile()

    return app
