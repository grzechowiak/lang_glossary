import pandas as pd
from config_paths import ConfigPaths#, BigQueryConfig
from config_datasets import ConfigDatasets

def validate_expected_columns_in_masters(
    dict_loaded: dict,
    dict_expected: dict
) -> bool:
    """
    Checks if loaded columns exactly match expected columns.

    Raises:
        ValueError: If columns are missing or unexpected.
    Returns:
        True if validation passes
    """

    loaded_cols = set(dict_loaded.keys())
    expected_cols = set(dict_expected.values())

    missing_columns = expected_cols - loaded_cols
    unexpected_columns = loaded_cols - expected_cols

    errors = []

    if missing_columns:
        errors.append(
            f"Missing expected columns: {', '.join(sorted(missing_columns))}"
        )

    if unexpected_columns:
        errors.append(
            f"Unexpected columns in file: {', '.join(sorted(unexpected_columns))}"
        )

    if errors:
        raise ValueError("\n".join(errors))

    return True


def load_csv_data(config: ConfigPaths, config_datasets: ConfigDatasets):
    """
    Load data from CSV files.

    Args:
        config: Configuration object with file paths
        config_datasets : Configuration for the datasets structure and naming

    Returns:
        Tuple of (original_sample_dict, bg_glossary_dict, ds_master_dict)
    """
    print("⏳ Loading data from CSV files...")

    ### 1. Load main dataset
    df_main = pd.read_csv(config.main_dataset, sep=config.csv_separator)
    df_sample = df_main.head(3) ## <---- define how rows will have the sample data on input
    original_sample_dict = df_sample.to_dict(orient='list')
    print(f"✅ Loaded sample dataset: {len(df_main)} rows")

    ### 2. Load master business glossary & rename columns
    df_glossary = pd.read_csv(config.master_glossary, sep=config.csv_separator)

    # Drop unwanted columns
    # cols_to_drop = [col for col in config_datasets.columns_to_drop if col in df_glossary.columns]
    # if cols_to_drop:
    #     df_glossary.drop(columns=cols_to_drop, inplace=True)
    df_glossary = df_glossary.rename(columns=config_datasets.column_mappings_master_bg)
    bg_glossary_dict = df_glossary.to_dict(orient='list')
    print(f"✅ Loaded master glossary: {len(df_glossary)} rows")

    ### 3. Load data stewards & rename columns
    df_stewards = pd.read_csv(config.data_stewards, sep=config.csv_separator)
    df_stewards = df_stewards.rename(columns=config_datasets.column_mappings_master_data_owners)
    ds_master_dict = df_stewards.to_dict(orient='list')
    print(f"✅ Loaded data stewards: {len(df_stewards)} rows")

    return original_sample_dict, bg_glossary_dict, ds_master_dict


def load_data(config: ConfigPaths, config_datasets: ConfigDatasets):
    """
    Load data based on config type.
    Simple router function.

    Args:
        config: Either Config or BigQueryConfig
        config_datasets : Configuration for the datasets structure and naming

    Returns:
        Tuple of (original_sample_dict, bg_glossary_dict, ds_master_dict)
    """
    # if isinstance(config, BigQueryConfig) and config.use_bigquery:
    #     return load_bigquery_data(config)
    # else:
    return load_csv_data(config, config_datasets)