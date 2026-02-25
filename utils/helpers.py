import pandas as pd
from typing import Iterable, List, Dict
from pydantic import BaseModel
from typing import Type
from datetime import datetime


def template_enricher(
    template_df: pd.DataFrame,
    enrich_df: pd.DataFrame,
    join_keys: List[str],
    fill_cols: Iterable[str],
) -> pd.DataFrame:
    """
    Enrich a template dataframe by left-joining another dataframe
    and filling selected columns with a clear priority:

        1) value from enrichment dataframe (master glossary)
        2) existing template value (if a value (non-place holder value) existed then we keep it)
        3) default placeholder (eg. <rag> or other tag with a template)


    Only columns listed in `fill_cols` are modified.
    All merge helper columns (_bg) are removed in the final output.
    """

    # Work on a copy to avoid mutating the input
    base = template_df.copy()

    # Decide placeholder (global, explicit, fast)
    default_placeholder = (
        "<rag>" if (base == "<rag>").any().any() else "<ds_master>"
    )

    # Left join enrichment data
    merged = base.merge(
        enrich_df,
        how="left",
        on=join_keys,
        suffixes=("", "_bg"),
    )

    # Enrich only selected columns
    for col in fill_cols:
        bg_col = f"{col}_bg"

        # Skip if the enrichment column doesn't exist
        if bg_col not in merged.columns or col not in merged.columns:
            continue

        merged[col] = (
            merged[bg_col]
            .combine_first(merged[col])
            .fillna(default_placeholder)
        )

    # Remove all enrichment-side columns
    bg_cols_to_drop = [c for c in merged.columns if c.endswith("_bg")]
    merged.drop(columns=bg_cols_to_drop, inplace=True)

    # Ensure no columns from enrich_df leak into the result
    merged = merged[template_df.columns]

    return merged


def check_columns_with_pydantic(df: pd.DataFrame, schema: Type[BaseModel]) -> bool:
    """
    Raises ValueError if DataFrame columns are not exactly the same set
    as the Pydantic model fields (missing or extra columns).
    """
    expected = set(schema.model_fields.keys())
    actual = set(df.columns)

    missing = sorted(expected - actual)
    extra = sorted(actual - expected)

    if missing or extra:
        raise ValueError(
            f"DataFrame columns do not match schema {schema.__name__}.\n"
            f"Missing: {missing}\n"
            f"Extra:   {extra}"
        )
    return True



def save_outputs(df_result, context_text, table_summary_text, cfg_paths):
    """
    Saving the final results of the Agent into separate files along with
    the timestamp
    """

    cfg_paths.output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    files = {
        "csv": cfg_paths.output_dir / f"{timestamp}_{cfg_paths.output_filename_suffix_final_table}.csv",
        "context": cfg_paths.output_dir / f"{timestamp}_{cfg_paths.output_filename_suffix_context_rag}.txt",
        "summary": cfg_paths.output_dir / f"{timestamp}_{cfg_paths.output_filename_suffix_table_summary}.txt",
    }

    # Save files
    df_result.to_csv(files["csv"], index=False, sep=cfg_paths.csv_separator)
    files["context"].write_text(context_text, encoding="utf-8")
    files["summary"].write_text(table_summary_text, encoding="utf-8")

    return files
