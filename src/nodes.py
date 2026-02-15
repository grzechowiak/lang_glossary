import json
import pandas as pd
from pathlib import Path
from langchain_openai import ChatOpenAI

# Internal Imports
from src.state import AgentState, TemplateOutput, ValidationResult
from rag.config_rag import RAGConfig
from rag.retriever_formatting import PrepareRetrieval
from utils.helpers import template_enricher
from config_paths import Config
from src.prompts import GENERATOR_PROMPT, VALIDATOR_PROMPT

config = Config()

# --- LLM Setup ---
gpt_model = config.llm_model
llm = ChatOpenAI(model=gpt_model, temperature=0)

structured_llm = llm.with_structured_output(TemplateOutput)
critic_llm = llm.with_structured_output(ValidationResult)


# --- NODE 1: Prepare Template ---
def prepare_template_node(state: AgentState) -> AgentState:
    """Prepare the Business Glossary template skeleton from the sampled dataset."""
    print("⏳ Fetching the template...")

    framework_def = state.get("framework_def")
    original_sample_dict = state.get("source_original_table")

    # Build the Framework
    df_template_v0 = pd.DataFrame(original_sample_dict)
    df_template_v1 = pd.DataFrame({
        "Table Name": framework_def['table_name_value'],
        "Column Name": df_template_v0.columns,
        "Sample Values": [df_template_v0[col].tolist() for col in df_template_v0.columns],
    })

    # Fill RAG columns with placeholder
    for c in framework_def['search_with_RAG']:
        df_template_v1[c] = "<agent>"

    # Fill data steward file columns with placeholder
    for c in framework_def['search_with_data_steward_file']:
        df_template_v1[c] = "<ds_master>"

    print("✅ Template is ready! Following structure will be filled in the next steps:")
    print(df_template_v1.head(5))

    return {"template_df": df_template_v1.to_dict(orient='list')}


# --- NODE 2: Fill Master Business Glossary ---
def fill_master_business_glossary_node(state: AgentState) -> AgentState:
    """Enrich the template from the Master Business Glossary."""
    print("\n⏳ Cross-checking with Master Business Glossary...")

    framework_def = state.get("framework_def")
    df_template = pd.DataFrame(state.get("template_df"))
    df_bg_glossary = pd.DataFrame(state.get("master_business_glossary"))

    fill_cols = framework_def['search_with_RAG'] + framework_def['additional_col']

    df_template_updated = template_enricher(
        template_df=df_template,
        enrich_df=df_bg_glossary,
        join_keys=["Table Name", "Column Name"],
        fill_cols=fill_cols,
    )

    return {"template_df": df_template_updated.to_dict(orient='list')}


# --- NODE 3: Fill Data Steward & Filter RAG ---
def fill_master_data_steward_node_and_rag_filter(state: AgentState) -> AgentState:
    """Enrich from Master Data Steward list and prepare RAG search list."""
    print("⏳ Cross-checking with Master Data Owner/Steward File...")

    ######################## PART 1/2 ########################
    # Perform the enrichment using Master Data Owner/Steward file
    framework_def = state.get("framework_def")
    df_template = pd.DataFrame(state.get("template_df"))
    df_do_master = pd.DataFrame(state.get("master_data_owner"))

    # Enrichment
    fill_cols = framework_def['search_with_data_steward_file']
    df_template_updated = template_enricher(
        template_df=df_template,
        enrich_df=df_do_master,
        join_keys=["Table Name"],
        fill_cols=fill_cols,
    )

    full_dict = df_template_updated.to_dict(orient='list')

    print("✅ Master files has been reviewed! Following structure will be sent to an Agent to generate missing fields:")
    print(df_template_updated.head(5))

    ######################## PART 2/2 ########################
    ### When full template is ready, filter only those rows which need to be filled in by RAG
    # (keep only rows with placeholders in RAG columns) - this will be used for retrieval and building
    # the context for the generator

    # Restrict RAG to only columns that still contain placeholders (extract rows which contains tag <...>)
    pattern = r"<.*?>"
    df_missing = df_template_updated[
        df_template_updated.apply(lambda row: row.astype(str).str.contains(pattern).any(), axis=1)
    ]

    ## Below object will be used to perform RAG searches (column name + sample values,
    # e.g. Account_number: [12345, 67890, 111213])
    dict_for_RAG_search = dict(zip(df_missing["Column Name"], df_missing["Sample Values"]))

    return {
        "template_df": full_dict, # in fact doesn't need to be retrieved but returned as updated object for clarity
        "entire_table_context": full_dict,
        "RAG_cols_with_samples": dict_for_RAG_search,
    }


# --- NODE 4: RAG Retrieval ---
def rag_retrieval_node(state: AgentState, project_root: Path) -> AgentState:
    """Perform retrieval and build context prompt."""
    col_samples = state.get("RAG_cols_with_samples")

    # Initialize RAG (assuming paths are relative to root where script is run)
    # basic_path = Path.cwd()
    cfg = RAGConfig(project_root=project_root)
    prep = PrepareRetrieval(cfg)

    results = prep.retrieve_for_all_columns(col_samples)
    company_context_prompt = prep.build_prompt_and_format(results)

    return {"RAG_company_context": company_context_prompt}


# --- NODE 5: Generator Agent ---
def generator_node(state: AgentState):
    """5. Define the Generator Agent logic"""

    iterations = state.get('iterations', 0) + 1
    print(f"--- GENERATOR: Filling the template (Attempt {iterations}) ---")

    # Check if this is a human-requested regeneration
    has_feedback = state.get('human_feedback') is not None

    if has_feedback:
        print(f"ℹ️ Generating content based on feedback: {state.get('human_feedback', '')[:50]}...")
    else:
        print("ℹ️ Generating initial content")

    try:
        # Bring context from state
        full_table_context = json.dumps(state.get('entire_table_context', {}), indent=2)
        rag_company_context = state.get('RAG_company_context', "No additional context provided.")

        # Prepare critic feedback if available
        critic_feedback = ""
        if state.get('error_message', "none") != "none":
            critic_feedback = f"\n\nCRITIC FEEDBACK FROM PREVIOUS ATTEMPT:\n{state.get('error_message')}\nPlease fix these issues."

        # Prepare human feedback if available
        human_feedback = ""
        if has_feedback:
            human_feedback = f"\n\nHUMAN FEEDBACK:\n{state.get('human_feedback')}"

            # Include any history of human modifications
            if state.get('human_review_history'):
                human_feedback += "\n\nPrevious human modifications:"
                for entry in state.get('human_review_history'):
                    human_feedback += f"\n- {entry}"

        # Format the prompt using LangChain's template
        formatted_messages = GENERATOR_PROMPT.format_messages(
            full_table_context=full_table_context,
            rag_company_context=rag_company_context,
            critic_feedback=critic_feedback,
            human_feedback=human_feedback
        )

        print("Generating content...")
        response = structured_llm.invoke(formatted_messages)
        print("✅ Content generated successfully")

        return {
            "result": response.columns,
            "iterations": iterations
        }
    except Exception as e:
        print(f"Error in generator: {e}")
        import traceback
        traceback.print_exc()

        # Return the original content if there's an error
        return {
            "result": state.get('result', []),
            "iterations": iterations,
            "error_message": f"Generator error: {str(e)}"
        }


# --- NODE 6: Validator Agent ---
def validator_node(state: AgentState):
    """6. Define the Validator Agent logic"""

    print("--- CRITIC: Reviewing... ---")

    # Check if human feedback is present
    has_human_feedback = state.get('human_feedback') is not None

    # If this is a human-requested regeneration, use simplified validation
    if has_human_feedback:
        print("✓ Simplified validation for human-guided changes")
        return {
            "error_message": "none",  # Pass validation
            "review_history_validator": [f"Validation simplified: Changes guided by human feedback"]
        }

    try:
        # Regular validation
        # Calculate counts to check for data loss
        input_table = state.get('entire_table_context', {})
        expected_count = len(input_table.get('Column Name', []))
        actual_count = len(state['result'])

        # Basic sanity check
        if actual_count == 0:
            return {
                "error_message": "No results generated.",
                "review_history_validator": [f"Validation failed: No results"]
            }

        if actual_count != expected_count:
            return {
                "error_message": f"Expected {expected_count} columns but got {actual_count}.",
                "review_history_validator": [f"Validation failed: Column count mismatch"]
            }

        current_work = "\n".join([
            f"Column: {c.column_name}\n"
            f"Proposed Description: {c.column_description}\n"
            f"Source Used: {c.extra__add_source_explained}\n"
            f"Evidence/Logic: {c.extra__add_citation_of_the_hit}\n"
            "---"
            for c in state['result']
        ])

        # Bring context from state
        rag_company_context = state.get('RAG_company_context', "No additional context provided.")
        full_table_context = state.get('entire_table_context', {})

        # Format the prompt using LangChain's template
        formatted_messages = VALIDATOR_PROMPT.format_messages(
            rag_company_context=rag_company_context,
            full_table_context=full_table_context,
            current_work=current_work,
            expected_count=expected_count,
            actual_count=actual_count
        )

        review = critic_llm.invoke(formatted_messages)

        if not review.is_valid:
            print(f"--- CRITIC FEEDBACK: {review.feedback} ---")
            return {
                "error_message": review.feedback,
                "review_history_validator": [f"Step {state['iterations']} Critic: {review.feedback}"]
            }

        print("✅ Validation passed")
        return {
            "error_message": "none",
            "review_history_validator": [f"Critique (Passed): {review.feedback}"]
        }
    except Exception as e:
        print(f"Error in validator: {e}")
        # If we've already tried a few times, just pass it to human review
        if state.get('iterations', 0) >= 2:
            return {
                "error_message": "none",
                "review_history_validator": [f"Validation error, but proceeding: {str(e)}"]
            }
        else:
            return {
                "error_message": f"Validation error: {str(e)}",
                "review_history_validator": [f"Validation error: {str(e)}"]
            }

# --- NODE 7: Human Review ---
from src.ui_helpers import display_with_gradio


def human_review_node(state: AgentState) -> AgentState:
    """Human review of the generated business glossary using Gradio."""

    print("\n" + "=" * 100)
    print("HUMAN REVIEW REQUIRED".center(100))
    print("=" * 100)

    # Get the generated content
    generated_content = state.get('result', [])

    # Summary statistics
    total_entries = len(generated_content)
    print(f"\nTotal entries: {total_entries}")
    print("\nOpening Gradio interface for review. Please make your selections there...")

    # Launch Gradio interface for review
    result = display_with_gradio(generated_content)

    # Process the result based on action type
    if "result" in result:
        # Directly using modified results from Gradio
        print("\n✅ Applied modifications directly from Gradio interface")

        # Convert the dict results back to ColumnDef objects
        from src.state import ColumnDef
        updated_columns = []
        for col_dict in result["result"]:
            updated_columns.append(ColumnDef(**col_dict))

        return {
            "result": updated_columns,
            "human_approved": True,
            "human_feedback": None,  # Clear feedback since we're approving
            "human_review_history": result.get("human_review_history", ["Modified directly in Gradio interface"])
        }

    elif result.get("human_approved", False):
        # User approved all entries
        print("\n✅ All entries approved.")
        return {
            "human_approved": True,
            "human_feedback": None,  # Clear feedback
            "human_review_history": result.get("human_review_history", ["Human reviewer approved all entries"])
        }

    else:
        # User requested regeneration
        feedback = result.get("human_feedback", "Please review and improve the business glossary entries")
        print(f"\n⟳ Requesting regeneration with feedback: {feedback}")
        print("The generator will create new content based on your feedback.")

        # Reset iterations counter to avoid hitting max too soon
        current_iterations = state.get('iterations', 0)

        return {
            "human_approved": False,
            "human_feedback": feedback,  # Set feedback for generator
            "error_message": None,  # Clear error message
            "iterations": max(0, current_iterations - 1),  # Give it more attempts
            "human_review_history": result.get("human_review_history",
                                               [f"Human reviewer requested regeneration: {feedback}"])
        }