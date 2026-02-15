import pandas as pd
from typing import Iterable, List, Dict


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



# def df_to_dict(df: pd.DataFrame) -> Dict[str, List[str]]:
#     """
#     Transform each column into a list of string values:
#       { "colA": ["v1","v2",...], "colB": [...], ... }
#     """
#     out: Dict[str, List[str]] = {}
#
#     for col in df.columns:
#         s = df[col].dropna()
#         out[col] = [str(v) for v in s.tolist()]
#
#     return out