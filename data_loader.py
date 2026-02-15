import pandas as pd
from config_paths import Config#, BigQueryConfig


def load_csv_data(config: Config):
    """
    Load data from CSV files.

    Args:
        config: Configuration object with file paths

    Returns:
        Tuple of (original_sample_dict, bg_glossary_dict, ds_master_dict)
    """
    print("⏳ Loading data from CSV files...")

    # Load main dataset
    df_main = pd.read_csv(config.main_dataset, sep=config.csv_separator)
    df_sample = df_main.head(3) ## <---- define how rows will have the sample data on input
    original_sample_dict = df_sample.to_dict(orient='list')
    print(f"✅ Loaded sample dataset: {len(df_main)} rows")

    # Load master business glossary
    df_glossary = pd.read_csv(config.master_glossary, sep=config.csv_separator)

    # Drop unwanted columns
    cols_to_drop = [col for col in config.columns_to_drop if col in df_glossary.columns]
    if cols_to_drop:
        df_glossary.drop(columns=cols_to_drop, inplace=True)

    bg_glossary_dict = df_glossary.to_dict(orient='list')
    print(f"✅ Loaded master glossary: {len(df_glossary)} rows")

    # Load data stewards
    df_stewards = pd.read_csv(config.data_stewards, sep=config.csv_separator)
    ds_master_dict = df_stewards.to_dict(orient='list')
    print(f"✅ Loaded data stewards: {len(df_stewards)} rows")

    return original_sample_dict, bg_glossary_dict, ds_master_dict


# def load_bigquery_data(config: BigQueryConfig):
#     """
#     Load data from BigQuery.
#     Use this when you migrate to GCP.
#
#     Args:
#         config: BigQuery configuration object
#
#     Returns:
#         Tuple of (original_sample_dict, bg_glossary_dict, ds_master_dict)
#     """
#     print("Loading data from BigQuery...")
#
#     try:
#         from google.cloud import bigquery
#
#         client = bigquery.Client(project=config.bq_project_id)
#
#         # Query main dataset (sample 3 rows)
#         query_main = f"""
#             SELECT *
#             FROM `{config.bq_project_id}.{config.bq_dataset}.{config.bq_main_table}`
#             LIMIT 3
#         """
#         df_main = client.query(query_main).to_dataframe()
#         print(f"✓ Loaded main dataset: {len(df_main)} rows")
#
#         # Query master glossary (exclude certain columns)
#         cols_to_exclude = ", ".join(config.columns_to_drop)
#         query_glossary = f"""
#             SELECT * EXCEPT ({cols_to_exclude})
#             FROM `{config.bq_project_id}.{config.bq_dataset}.{config.bq_glossary_table}`
#         """
#         df_glossary = client.query(query_glossary).to_dataframe()
#         print(f"✓ Loaded master glossary: {len(df_glossary)} rows")
#
#         # Query data stewards
#         query_stewards = f"""
#             SELECT *
#             FROM `{config.bq_project_id}.{config.bq_dataset}.{config.bq_stewards_table}`
#         """
#         df_stewards = client.query(query_stewards).to_dataframe()
#         print(f"✓ Loaded data stewards: {len(df_stewards)} rows")
#
#         return (
#             df_main.to_dict(orient='list'),
#             df_glossary.to_dict(orient='list'),
#             df_stewards.to_dict(orient='list')
#         )
#
#     except ImportError:
#         print("Error: google-cloud-bigquery not installed")
#         print("   Install with: pip install google-cloud-bigquery")
#         raise
#     except Exception as e:
#         print(f"Error loading from BigQuery: {e}")
#         raise


def load_data(config: Config):
    """
    Load data based on config type.
    Simple router function.

    Args:
        config: Either Config or BigQueryConfig

    Returns:
        Tuple of (original_sample_dict, bg_glossary_dict, ds_master_dict)
    """
    # if isinstance(config, BigQueryConfig) and config.use_bigquery:
    #     return load_bigquery_data(config)
    # else:
    return load_csv_data(config)