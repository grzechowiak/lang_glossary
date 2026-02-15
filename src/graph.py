from langgraph.graph import StateGraph, END
from src.state import AgentState
from src.nodes import (
    prepare_template_node,
    fill_master_business_glossary_node,
    fill_master_data_steward_node_and_rag_filter,
    rag_retrieval_node,
    generator_node,
    validator_node
)
from functools import partial
from pathlib import Path

def router(state: AgentState):
    """Determines next step based on validation results."""
    if state["error_message"] == "none":
        return "end"
    if state["iterations"] >= 3:
        print("!!! Max attempts reached. Ending with current result.")
        return "end"
    return "generate"

def build_graph(project_root: Path):
    """Constructs and compiles the StateGraph."""
    workflow = StateGraph(AgentState)

    rag_node_with_path = partial(rag_retrieval_node, project_root=project_root)

    # Add Nodes
    workflow.add_node("prepare_template", prepare_template_node)
    workflow.add_node("fill_master_business_glossary", fill_master_business_glossary_node)
    workflow.add_node("fill_master_data_steward", fill_master_data_steward_node_and_rag_filter)
    workflow.add_node("RAG_retrieve", rag_node_with_path)
    workflow.add_node("generate", generator_node)
    workflow.add_node("validate", validator_node)

    # Set Entry Point
    workflow.set_entry_point("prepare_template")

    # Add Edges
    workflow.add_edge("prepare_template", "fill_master_business_glossary")
    workflow.add_edge("fill_master_business_glossary", "fill_master_data_steward")
    workflow.add_edge("fill_master_data_steward", "RAG_retrieve")
    workflow.add_edge("RAG_retrieve", "generate")
    workflow.add_edge("generate", "validate")

    # Add Conditional Edges
    workflow.add_conditional_edges(
        "validate",
        router,
        {
            "generate": "generate",
            "end": END
        }
    )

    return workflow.compile()

# # Visualize the new nodes structure
# try:
#     display(Image(app.get_graph().draw_mermaid_png()))
# except Exception as e:
#     print("Graph compilation failed:", str(e))