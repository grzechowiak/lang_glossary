# data_steward_agent/
# │
# ├── .env                       # Your OpenAI API Key
# ├── main.py                    # Entry point to run the app
# ├── requirements.txt           # Dependencies
# │
# ├── data/                      # Folder for your input CSVs
# │   ├── datasets/              # (Place dataset_csv.csv here)
# │   ├── master_business_glossary/
# │   └── stewards_and_owners/
# │
# ├── rag/                       # (Keep your existing RAG folder here)
# ├── utils/                     # (Keep your existing utils folder here)
# │
# └── src/                       # NEW: Modular code folder
#     ├── __init__.py
#     ├── state.py               # Data models and State definition
#     ├── nodes.py               # The core logic (Generator, Validator, etc.)
#     └── graph.py               # Graph definition and compilation

import os
import datetime
import pandas as pd
from pathlib import Path

# Load API Keys
from dotenv import load_dotenv
load_dotenv()
if not os.environ.get("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY not found in environment variables.")

# Import the graph builder
from src.graph import build_graph

# Setup basic paths
BASIC_PATH = Path.cwd()
DATA_FULL_PATH = BASIC_PATH / "data/"
SAVE_TEMP_RESULTS = BASIC_PATH / "99_playground"

# Ensure output directory exists
SAVE_TEMP_RESULTS.mkdir(parents=True, exist_ok=True)


def load_initial_data():
    """Reads CSVs and prepares the initial state dictionary."""
    print("Loading datasets...")

    # Define file names
    main_dataset = 'datasets/dataset_csv.csv'
    master_bg = 'master_business_glossary/master_business_glossary_csv.csv'
    master_ds = 'stewards_and_owners/data_stewards.csv'

    # Read Main Dataset
    df = pd.read_csv(DATA_FULL_PATH / main_dataset, sep=';')
    df_sample = df.head(3)
    original_sample_dict = df_sample.to_dict(orient='list')

    # Read Master Business Glossary
    df_bg = pd.read_csv(DATA_FULL_PATH / master_bg, sep=';')
    # Drops columns to be discussed (as per notebook)
    cols_to_drop = [
        "Data Steward Approval", "Data Steward Feedback (only if not approved)",
        "Data Owner Approval", "Data Owner Feedback (only if not approved)"
    ]
    df_bg.drop(columns=[c for c in cols_to_drop if c in df_bg.columns], inplace=True)
    bg_glossary_dict = df_bg.to_dict(orient='list')

    # Read Master Data Steward
    df_ds = pd.read_csv(DATA_FULL_PATH / master_ds, sep=';')
    ds_master_dict = df_ds.to_dict(orient='list')

    return original_sample_dict, bg_glossary_dict, ds_master_dict


def get_framework_config():
    """Defines the configuration for the template."""
    return {
        "table_name_value": 'client_account',
        "additional_col": ["Sample Values"],
        "search_with_RAG": [
            'Business Domain Name', 'Business Sub-Domain Name', 'Business Name',
            'Column Description', 'Attribute related business rationale',
            'Attribute logical business rules'
        ],
        "search_with_data_steward_file": [
            'Data Owner Name', 'Data Owner E-Mail'
        ],
    }


def main():
    # 1. Prepare Data
    try:
        sample_dict, bg_dict, ds_dict = load_initial_data()
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Please ensure 'data/' directory contains the required CSV files.")
        return

    # 2. Initialize State
    initial_state = {
        "framework_def": get_framework_config(),
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

    # 3. Run Graph
    app = build_graph(project_root=BASIC_PATH)
    print("--- Starting Workflow ---")

    final_output = app.invoke(initial_state, {"recursion_limit": 10})

    # 4. Process Results
    if final_output and 'result' in final_output:
        df_result = pd.DataFrame([c.model_dump() for c in final_output['result']])

        # Save
        datestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_file = SAVE_TEMP_RESULTS / f'BG_TEST_v1_{datestamp}.csv'
        df_result.to_csv(output_file, index=False, sep=';')

        print(f"\nSuccess! File saved to: {output_file}")
        print("\n" + "=" * 30)
        print(df_result.head())
    else:
        print("Workflow finished but no result was generated.")


if __name__ == "__main__":
    main()