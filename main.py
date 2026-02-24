import os
import datetime
import pandas as pd
# from pathlib import Path

# Load environment variables
from dotenv import load_dotenv

load_dotenv(override=True)
if not os.environ.get("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY not found")
    exit(1)

# Import loaders / helpers
from utils.data_loader import load_data
from utils.data_loader import validate_expected_columns_in_masters
from utils.helpers import save_outputs

# Import the graph builder
from src.graph import build_graph

# Import Configs
from configs.config_paths import ConfigPaths
from rag.config_rag import RAGConfig
from configs.config_datasets import ConfigDatasets
from configs.config_agent import ConfigAgents
from src.state import TemplateOutput

def main():
    """Main execution flow - simple and clean."""

    print("=" * 60)
    print("BUSINESS GLOSSARY FILLING - AGENTIC WORKFLOW")
    print("=" * 60)

    # 1. Setup configuration
    cfg_paths = ConfigPaths()
    cfg_rag = RAGConfig(project_root=cfg_paths.project_root)
    cfg_datasets = ConfigDatasets()
    cfg_agents = ConfigAgents()

    print(f"\nProject root: {cfg_paths.project_root}")
    print(f"Embedding model: {cfg_rag.embedding_model}")
    print(f"LLM model: {cfg_agents.llm_model}\n")

    # 2. Load data
    try:
        sample_dict, bg_dict, ds_dict = load_data(cfg_paths, cfg_datasets)
    except FileNotFoundError as e:
        print(f"\n Error: {e}")
        print("\nMake sure these files exist:")
        print(f"  - {cfg_paths.main_dataset}")
        print(f"  - {cfg_paths.master_glossary}")
        print(f"  - {cfg_paths.data_stewards}")
        return 1
    except Exception as e:
        print(f"\n Error loading data: {e}")
        return 1

    # 3. Validate if files contains columns which are expected
    bg_masters_columns = cfg_datasets.column_mappings_master_bg
    ds_master_columns = cfg_datasets.column_mappings_master_data_owners
    validate_expected_columns_in_masters(bg_dict, bg_masters_columns)
    validate_expected_columns_in_masters(ds_dict, ds_master_columns)

    # 4. Setup initial state
    initial_state = {
        "framework_def": cfg_datasets.get_framework_dict(),
        "source_original_table": sample_dict,
        "master_business_glossary": bg_dict,
        "master_data_owner": ds_dict,
        "RAG_cols_with_samples": {},
        "RAG_company_context": "",
        "entire_table_context": {},
        "template_df": {},
        "result": [],
        "error_message": "",
        "iterations": 0,
        "review_history_validator": []
    }

    # 5. Run workflow
    print("\n" + "=" * 60)
    print("STARTING WORKFLOW")
    print("=" * 60 + "\n")

    app = build_graph(project_root=cfg_paths.project_root)

    try:
        final_output = app.invoke(initial_state, {"recursion_limit": cfg_agents.recursion_limit})
    except Exception as e:
        print(f"\n Workflow failed: {e}")
        return 1

    # 6. Save results
    if not final_output or 'result' not in final_output or not final_output['result']:
        print("\n No results generated")
        return 1

    # Unpack results
    result_fetch: TemplateOutput = final_output["result"]
    df_result = pd.DataFrame([c.model_dump() for c in result_fetch.rows])
    context_txt = final_output['RAG_company_context']
    df_table_summary = result_fetch.table_summary

    # Save
    save_outputs(df_result = df_result,
                 context_text = context_txt,
                 table_summary_text = df_table_summary,
                 cfg_paths = cfg_paths)

    # # Generate filename
    # timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # filename = f"{timestamp}_{cfg_paths.output_filename_suffix}.csv"
    # filename_context_rag = f"{timestamp}_{cfg_paths.output_filename_suffix_context_rag}.txt"
    # filename_table_summary = f"{timestamp}_{cfg_paths.output_filename_suffix_table_summary}.txt"
    #
    # output_file = cfg_paths.output_dir / filename
    # output_file2 = cfg_paths.output_dir / filename_context_rag
    # output_file3 = cfg_paths.output_dir / filename_table_summary
    #
    # # Save
    # df_result.to_csv(output_file, index=False, sep=cfg_paths.csv_separator) # final table
    # output_file2.write_text(context_txt, encoding="utf-8") # context
    # output_file3.write_text(df_table_summary, encoding="utf-8")  # table_summary

    # 7. Display results
    print("\n ✅ Agents Finished!")
    print(f"\n📝 Saved to: {cfg_paths.output_dir}")
    print(f"Columns processed: {len(df_result)}")
    print(f"Iterations: {final_output.get('iterations', 0)}")

    print("\n" + "-" * 60)
    print("PREVIEW:")
    print("-" * 60)
    print(df_result)
    print("-" * 60)

    return 0

if __name__ == "__main__":
    exit(main())