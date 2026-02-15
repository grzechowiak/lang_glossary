from langgraph.graph import StateGraph, END
from src.state import AgentState
from src.nodes import (
    prepare_template_node,
    fill_master_business_glossary_node,
    fill_master_data_steward_node_and_rag_filter,
    rag_retrieval_node,
    generator_node,
    validator_node,
    human_review_node
)
from functools import partial
from pathlib import Path


def should_continue_generating(state: AgentState) -> str:
    """Simple decision function for after validation."""
    error_message = state.get("error_message", "none")
    iterations = state.get("iterations", 0)

    if error_message == "none":
        print("✅ Validation passed, going to human review")
        return "human_review"
    elif iterations >= 3:
        print("⚠️ Max iterations reached, going to human review anyway")
        return "human_review"
    else:
        print(f"❌ Validation failed (iteration {iterations}), regenerating")
        return "generate"


def human_review_decision(state: AgentState) -> str:
    """Simple decision function for after human review."""
    human_approved = state.get("human_approved", False)

    if human_approved:
        print("✅ Human approved, ending workflow")
        return "end"
    else:
        print("⟳ Human requested regeneration, going back to generator")
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
    workflow.add_node("human_review", human_review_node)

    # Set Entry Point
    workflow.set_entry_point("prepare_template")

    # Add Edges - strict linear flow for initial steps
    workflow.add_edge("prepare_template", "fill_master_business_glossary")
    workflow.add_edge("fill_master_business_glossary", "fill_master_data_steward")
    workflow.add_edge("fill_master_data_steward", "RAG_retrieve")
    workflow.add_edge("RAG_retrieve", "generate")
    workflow.add_edge("generate", "validate")  # Always validate after generating

    # Add Conditional Edges for decision points
    workflow.add_conditional_edges(
        "validate",
        should_continue_generating,
        {
            "human_review": "human_review",
            "generate": "generate"
        }
    )

    workflow.add_conditional_edges(
        "human_review",
        human_review_decision,
        {
            "end": END,
            "generate": "generate"
        }
    )

    return workflow.compile()

# # Visualize the new nodes structure
# try:
#     display(Image(app.get_graph().draw_mermaid_png()))
# except Exception as e:
#     print("Graph compilation failed:", str(e))