import json
import pandas as pd
from pathlib import Path
from langchain_openai import ChatOpenAI

# Internal Imports
from src.state import AgentState, TemplateOutput, ValidationResult
from rag.config_rag import RAGConfig
from rag.retriever_formatting import PrepareRetrieval
from utils.helpers import template_enricher
# from config_paths import ConfigPaths
from config_agent import ConfigAgents
from src.prompts import GENERATOR_PROMPT, VALIDATOR_PROMPT
from config_datasets import ConfigDatasets

# --- LLM Setup ---
# config = ConfigPaths()
cfg_agent = ConfigAgents()
gpt_model = cfg_agent.llm_model
cfg_dataset = ConfigDatasets()

llm = ChatOpenAI(model=gpt_model, temperature=0)
structured_llm = llm.with_structured_output(TemplateOutput)
critic_llm = llm.with_structured_output(ValidationResult)


# --- NODE 1: Prepare Template ---
def prepare_template_node(state: AgentState) -> AgentState:
    """Prepare the Business Glossary template skeleton from the sampled dataset."""
    print("⏳ Fetching the template...")

    original_sample_dict = state["source_original_table"]

    df_template = cfg_dataset.build_template(original_sample_dict)

    return {"template_df": df_template.to_dict(orient="list")}

    # framework_def = state.get("framework_def")
    # original_sample_dict = state.get("source_original_table")
    #
    # # Build the Framework
    # df_template_v0 = pd.DataFrame(original_sample_dict)
    # df_template_v1 = pd.DataFrame({
    #     "bucket_name": framework_def['bucket_name'],
    #     "dataset_name": framework_def['dataset_name'] ,
    #     "table_name": framework_def['table_name_value'],
    #     "column_name": df_template_v0.columns,
    #     "sample_values": [df_template_v0[col].tolist() for col in df_template_v0.columns],
    # })
    #
    # # Fill RAG columns with placeholder
    # for c in framework_def['search_with_RAG']:
    #     df_template_v1[c] = "<agent>"
    #
    # # Fill data steward file columns with placeholder
    # for c in framework_def['search_with_data_steward_file']:
    #     df_template_v1[c] = "<ds_master>"
    #
    # print("✅ Template is ready! Following structure will be filled in the next steps:")
    # print(df_template_v1.head(5))
    #
    # return {"template_df": df_template_v1.to_dict(orient='list')}


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
        join_keys=["bucket_name", "dataset_name", 'table_name', 'column_name'],
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
        join_keys=["bucket_name", "dataset_name", 'table_name'], # stewards are joined on table level
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
    dict_for_RAG_search = dict(zip(df_missing["column_name"], df_missing["sample_values"]))

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
    cfg = RAGConfig(project_root=project_root)
    prep = PrepareRetrieval(cfg)

    results = prep.retrieve_for_all_columns(col_samples)
    company_context_prompt = prep.build_prompt_and_format(results)

    return {"RAG_company_context": company_context_prompt}


# --- NODE 5: Generator Agent ---
def generator_node(state: AgentState):
    """5. Define the Generator Agent logic"""

    print(f"--- GENERATOR: Filling the template (Attempt {state['iterations'] + 1}) ---")

    # Bring context from state
    full_table_context = json.dumps(state.get('entire_table_context'), indent=2) # Convert dict to JSON on the fly so it's better formatted when supplying to the Agent
    rag_company_context = (state.get('RAG_company_context', "No additional context provided."))

    # Prepare critic feedback if available
    critic_feedback = ""
    if state['error_message'] != "none":
        critic_feedback = f"\n\nCRITIC FEEDBACK FROM PREVIOUS ATTEMPT:\n{state['error_message']}\nPlease fix these issues."

    # Format the prompt using LangChain's template
    formatted_messages = GENERATOR_PROMPT.format_messages(
        full_table_context=full_table_context,
        rag_company_context=rag_company_context,
        critic_feedback=critic_feedback
        )

    response = structured_llm.invoke(formatted_messages)
    return {
        "result": response.columns,
        "iterations": state['iterations'] + 1
    }


# --- NODE 6: Validator Agent ---
def validator_node(state: AgentState):
    """6. Define the Validator Agent logic"""

    print("--- CRITIC: Reviewing... ---")

    # Calculate counts to check for data loss
    input_table = state.get('entire_table_context', {})
    expected_count = len(input_table.get('Column Name', []))
    actual_count = len(state['result'])

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

    # if false (i.e. not valid)
    if not review.is_valid:
        print(f"--- CRITIC FEEDBACK: {review.feedback} ---")
        return {
            "error_message": review.feedback,
            "review_history_validator": [f"Step {state['iterations']} Critic: {review.feedback}"]
        }

    return {
        "error_message": "none",
        "review_history_validator": [f"Critique (Passed): {review.feedback}"]
    }