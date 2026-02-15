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

# def get_masked_key(variable_name):
#     key = os.getenv(variable_name)
#     if not key:
#         return None
#     # Show first 4 and last 4 characters, mask the middle
#     if len(key) > 8:
#         return f"{key[:4]}{'*' * 8}{key[-4:]}"
#     return "****" # Fallback if key is strangely short
#
# # 2. Validation and verification
# api_key = os.getenv("OPENAI_API_KEY")
#
# if not api_key:
#     print("❌ Error: OPENAI_API_KEY not found in environment.")
#     exit(1)
# else:
#     print(f"✅ API Key loaded: {get_masked_key('OPENAI_API_KEY')}")

# Import simple config and loader
from config_paths import Config
from data_loader import load_data

# Import the graph builder
from src.graph import build_graph

# Import RAG config
from rag.config_rag import RAGConfig


def main():
    """Main execution flow - simple and clean."""

    print("=" * 60)
    print("BUSINESS GLOSSARY FILLING - AGENTIC WORKFLOW")
    print("=" * 60)

    # 1. Setup configuration
    config = Config()
    rag_default_config = RAGConfig(project_root=config.project_root)

    # For BigQuery, use this instead:
    # from config_agent import BigQueryConfig
    # config = BigQueryConfig(project_root=Path.cwd())
    # config.bq_project_id = "your-project"
    # config.bq_dataset = "your-dataset"

    print(f"\nProject root: {config.project_root}")
    print(f"Embedding model: {rag_default_config.embedding_model}")
    print(f"LLM model: {config.llm_model}\n")

    # 2. Load data
    try:
        sample_dict, bg_dict, ds_dict = load_data(config)
    except FileNotFoundError as e:
        print(f"\n Error: {e}")
        print("\nMake sure these files exist:")
        print(f"  - {config.main_dataset}")
        print(f"  - {config.master_glossary}")
        print(f"  - {config.data_stewards}")
        return 1
    except Exception as e:
        print(f"\n Error loading data: {e}")
        return 1

    # 3. Setup initial state
    initial_state = {
        "framework_def": config.get_framework_dict(),
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

    # 4. Run workflow
    print("\n" + "=" * 60)
    print("STARTING WORKFLOW")
    print("=" * 60 + "\n")

    app = build_graph(project_root=config.project_root)

    try:
        final_output = app.invoke(initial_state, {"recursion_limit": config.recursion_limit})
    except Exception as e:
        print(f"\n Workflow failed: {e}")
        return 1

    # 5. Save results
    if not final_output or 'result' not in final_output or not final_output['result']:
        print("\n No results generated")
        return 1

    df_result = pd.DataFrame([c.model_dump() for c in final_output['result']])

    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename
    if config.include_timestamp:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{config.output_filename_prefix}_{timestamp}.csv"
    else:
        filename = f"{config.output_filename_prefix}.csv"

    output_file = config.output_dir / filename

    # Save
    df_result.to_csv(output_file, index=False, sep=config.csv_separator)

    # 6. Display results
    print("\n ✅ Agents Finished!")
    print(f"\n📝 Saved to: {output_file}")
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