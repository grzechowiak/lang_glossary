from __future__ import annotations

import io
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import HTTPException

from llm import regenerate_row_with_llm
from src.state import ColumnDefOutput


from pathlib import Path
import sys
import os
from datetime import datetime
# Add the project root to the Python path to import from configs
sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config_paths import ConfigPaths
# Initialize paths configuration
config_paths = ConfigPaths()


# The canonical column order comes from Pydantic model fields
CANONICAL_COLUMNS: List[str] = list(ColumnDefOutput.model_fields.keys())


def _coerce_sample_values(value: Any) -> str:
    """
    Normalize sample values to a string.

    Since the input is expected to be a string already (from the Pydantic model),
    this function mainly handles None/empty values and does basic string cleaning.
    """
    # Handle None or empty cases
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""

    # For strings (the expected case), just clean it up
    if isinstance(value, str):
        return value.strip()

    # For any unexpected type (legacy data, etc.), convert to string
    # This is just a safety measure, should rarely be needed
    return str(value)


def dataframe_from_csv_bytes(raw: bytes) -> pd.DataFrame:
    """
    Parse CSV bytes -> validate rows via Pydantic -> return DataFrame
    """
    try:
        df = pd.read_csv(io.BytesIO(raw), sep=';')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {e}")

    # Ensure all expected columns exist
    missing = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"CSV is missing columns: {missing}")

    # Keep only canonical columns and in canonical order
    df = df[CANONICAL_COLUMNS].copy()

    # Normalize NaN to None for validation
    records: List[Dict[str, Any]] = df.where(pd.notnull(df), None).to_dict(orient="records")

    validated_rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for idx, row in enumerate(records):
        # normalize sample_values
        row["sample_values"] = _coerce_sample_values(row.get("sample_values"))

        try:
            model = ColumnDefOutput(**row)
            validated_rows.append(model.model_dump())
        except Exception as e:
            errors.append({"row_index": idx, "error": str(e)})

    if errors:
        # return first few errors for readability
        raise HTTPException(status_code=400, detail={"message": "CSV validation failed", "errors": errors[:10]})

    return pd.DataFrame(validated_rows, columns=CANONICAL_COLUMNS)


# ---- Domain classes ----
@dataclass
class Session:
    df: pd.DataFrame
    status: str = "awaiting_action"  # or "accepted"
    change_log: List[Dict[str, Any]] = field(default_factory=list)

    def ensure_not_accepted(self) -> None:
        if self.status == "accepted":
            raise HTTPException(status_code=400, detail="Session already accepted (locked)")

    def to_rows(self) -> List[Dict[str, Any]]:
        return self.df.where(pd.notnull(self.df), None).to_dict(orient="records")

    def get_csv_content(self) -> str:
        return self.df.to_csv(index=False, sep=config_paths.csv_separator)

    def save_csv_to_file(self, filename_base: str = "Component_2_table") -> str:
        """
        Save the session dataframe to a CSV file using the configured paths.

        Args:
            filename_base: Base name for the output file (default: Component_2_table)

        Returns:
            Path to the saved file
        """
        # Ensure output directory exists
        os.makedirs(config_paths.output_dir, exist_ok=True)

        # Create filename with timestamp if configured
        if config_paths.include_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            full_filename = f"{filename_base}_{timestamp}.csv"
        else:
            full_filename = f"{filename_base}.csv"

        # Full path for output file
        output_path = config_paths.output_dir / full_filename

        # Save DataFrame to CSV using the configured separator
        self.df.to_csv(
            output_path,
            sep=config_paths.csv_separator,
            index=False
        )

        print(f"Saved CSV file to: {output_path}")
        return str(output_path)


class SessionStore:
    def __init__(self) -> None:
        self._sessions: Dict[str, Session] = {}

    def create_from_df(self, df: pd.DataFrame) -> str:
        sid = str(uuid.uuid4())
        self._sessions[sid] = Session(df=df)
        return sid

    def get(self, sid: str) -> Session:
        s = self._sessions.get(sid)
        if not s:
            raise HTTPException(status_code=404, detail="Unknown session_id")
        return s


store = SessionStore()


# ---- Domain operations ----
def edit_cell(sid: str, row: int, col: str, value: str) -> None:
    session = store.get(sid)
    session.ensure_not_accepted()

    df = session.df
    if col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Unknown column: {col}")
    if row < 0 or row >= len(df):
        raise HTTPException(status_code=400, detail=f"Row out of range: {row}")

    old = df.at[row, col]
    df.at[row, col] = value
    session.change_log.append({"type": "edit_cell", "row": row, "col": col, "old": old, "new": value})


def regenerate_row(sid: str, row: int, feedback: str) -> Dict[str, object]:
    session = store.get(sid)
    session.ensure_not_accepted()

    df = session.df
    if row < 0 or row >= len(df):
        raise HTTPException(status_code=400, detail=f"Row out of range: {row}")

    cols = list(df.columns)
    current_row = df.iloc[row].where(pd.notnull(df.iloc[row]), None).to_dict()

    try:
        candidate = regenerate_row_with_llm(cols, current_row, feedback)
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    # Soft-merge
    merged = dict(current_row)
    for c in cols:
        if c in candidate:
            merged[c] = candidate[c]

    # Normalize sample_values again (model might return string)
    merged["sample_values"] = _coerce_sample_values(merged.get("sample_values"))

    # Validate regenerated row via Pydantic (important!)
    try:
        validated = ColumnDefOutput(**merged).model_dump()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Regenerated row failed validation: {e}")

    df.loc[row, cols] = [validated[c] for c in cols]
    session.change_log.append({"type": "regen_row", "row": row, "feedback": feedback, "old": current_row, "new": validated})
    return validated


def accept(sid: str) -> None:
    """
    Mark a session as accepted (which locks it from further edits).
    """
    session = store.get(sid)
    session.ensure_not_accepted()
    session.status = "accepted"
    session.change_log.append({"type": "accept"})
    print(f"Session {sid} marked as accepted")