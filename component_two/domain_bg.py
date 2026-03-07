from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List
from datetime import datetime

import pandas as pd
from fastapi import HTTPException

from component_two.llm import regenerate_row_with_llm
from component_one.src.state import ColumnDefOutput
from component_two.utils.data_utils import DataUtils


@dataclass
class Session:
    """
    Represents a user editing session for a CSV dataset.

    Maintains the dataframe being edited, tracks status (whether edits are still allowed),
    and maintains a history of all changes made during the session.
    """
    df: pd.DataFrame
    status: str = "awaiting_action"  # or "accepted"
    change_log: List[Dict[str, Any]] = field(default_factory=list)

    def ensure_not_accepted(self) -> None:
        """
        Verify that the session is not in an accepted state.

        Raises HTTPException if session is already accepted.
        """
        if self.status == "accepted":
            raise HTTPException(status_code=400, detail="Session already accepted (locked)")

    def to_rows(self) -> List[Dict[str, Any]]:
        """Convert the session's DataFrame to a list of dictionaries (one per row)."""
        return self.df.where(pd.notnull(self.df), None).to_dict(orient="records")



class SessionManager:
    """
    Manages creation, retrieval, and operations on editing sessions.
    """

    def __init__(self) -> None:
        self._sessions: Dict[str, Session] = {}

    def create_from_df(self, df: pd.DataFrame) -> str:
        """
        Create a new session from a DataFrame.

        Args:
            df: Pandas DataFrame containing the data for the session

        Returns:
            str: Unique session ID
        """
        sid = str(uuid.uuid4())
        self._sessions[sid] = Session(df=df)
        return sid

    def get(self, sid: str) -> Session:
        """
        Retrieve a session by ID.

        Args:
            sid: Session ID

        Returns:
            Session object

        Raises:
            HTTPException: If session ID doesn't exist
        """
        s = self._sessions.get(sid)
        if not s:
            raise HTTPException(status_code=404, detail="Unknown session_id")
        return s

    def edit_cell(self, sid: str, row: int, col: str, value: str) -> None:
        """
        Edit a specific cell in the session's DataFrame.

        Args:
            sid: Session ID
            row: Row index
            col: Column name
            value: New cell value

        Raises:
            HTTPException: If session doesn't exist, is locked, or coordinates are invalid
        """
        session = self.get(sid)
        session.ensure_not_accepted()

        df = session.df
        if col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Unknown column: {col}")
        if row < 0 or row >= len(df):
            raise HTTPException(status_code=400, detail=f"Row out of range: {row}")

        old = df.at[row, col]
        df.at[row, col] = value
        session.change_log.append({"type": "edit_cell", "row": row, "col": col, "old": old, "new": value})

    def regenerate_row(self, sid: str, row: int, feedback: str) -> Dict[str, object]:
        """
        Regenerate a row using LLM based on user feedback.

        Args:
            sid: Session ID
            row: Row index
            feedback: User's natural language feedback on how to modify the row

        Returns:
            Dict containing the validated row data after regeneration

        Raises:
            HTTPException: If session doesn't exist, is locked, row is invalid,
                          or LLM generation fails
        """
        session = self.get(sid)
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
        merged["sample_values"] = DataUtils.coerce_sample_values(merged.get("sample_values"))

        # Validate regenerated row via Pydantic (important!)
        try:
            validated = ColumnDefOutput(**merged).model_dump()
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Regenerated row failed validation: {e}")

        df.loc[row, cols] = [validated[c] for c in cols]
        session.change_log.append(
            {"type": "regen_row", "row": row, "feedback": feedback, "old": current_row, "new": validated})
        return validated

    def accept(self, sid: str) -> str:
        """
    Mark a session as accepted, locking it from further edits.
    Automatically saves the data to a CSV file as part of the acceptance process.

    Performs final validation to ensure all data meets schema requirements.

    Args:
        sid: Session ID

    Returns:
        Path to the saved CSV file

    Raises:
        HTTPException: If session doesn't exist, is already accepted, or contains invalid data
    """
        session = self.get(sid)
        session.ensure_not_accepted()

        # Perform final validation of the dataframe before accepting
        try:
            validated_df = DataUtils.validate_dataframe(session.df)
            session.df = validated_df
        except HTTPException as e:
            if isinstance(e.detail, dict) and "message" in e.detail:
                e.detail["message"] = f"Session acceptance failed: {e.detail['message']}"
            raise e

        # Change status to accepted
        session.status = "accepted"

        # Save to CSV as part of acceptance using the helper function
        filename_base = f"accepted_{sid[:8]}"
        file_path = DataUtils.save_dataframe_to_csv(session.df, filename_base)

        # Log the acceptance and save action
        session.change_log.append({
            "type": "accept",
            "timestamp": datetime.now().isoformat(),
            "saved_path": file_path,
            "validation": "passed"
        })

        print(f"Session {sid} marked as accepted and saved to {file_path}")
        return file_path


session_manager = SessionManager()