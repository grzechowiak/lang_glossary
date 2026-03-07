import sys
import os

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel, Field

import domain_bg as domain
from component_two.utils.data_utils import DataUtils

from datetime import datetime

timestamp = datetime.now().strftime("DATE:_%Y-%m-%d_TIME:_%H-%M-%S")


class EditCellRequest(BaseModel):
    """Request model for cell editing operations."""
    row: int = Field(..., ge=0)
    col: str = Field(..., min_length=1)
    value: str = Field(..., min_length=1)


class RegenerateRowRequest(BaseModel):
    """Request model for row regeneration operations."""
    feedback: str = Field(..., min_length=1)


bg_app = FastAPI(title=f'Session-based Table Review API {timestamp}')


@bg_app.post("/sessions")
async def create_session(csv: UploadFile = File(...)):
    """
    Create a new editing session from an uploaded CSV file.

    The CSV file is validated against the expected schema before creating the session.
    """
    raw = await csv.read()
    df = DataUtils.load_csv_and_validate(raw)
    sid = domain.session_manager.create_from_df(df)  # Updated from store to session_manager
    return {"session_id": sid, "status": "awaiting_action", "row_count": int(len(df))}


@bg_app.patch("/sessions/{sid}/cell_modification")
def patch_cell(sid: str, req: EditCellRequest):
    """
    Edit a specific cell in a session's data.

    Returns the updated session data.
    """
    domain.session_manager.edit_cell(sid, req.row, req.col, req.value)  # Updated to use session_manager
    s = domain.session_manager.get(sid)  # Updated to use session_manager
    return {"ok": True, "session_id": sid, "status": s.status, "rows": s.to_rows()}


@bg_app.post("/sessions/{sid}/rows/{row}/row_regenerate")
def regen_row(sid: str, row: int, req: RegenerateRowRequest):
    """
    Regenerate a row using LLM based on user feedback.

    Returns the updated row data.
    """
    new_row = domain.session_manager.regenerate_row(sid, row, req.feedback.strip())  # Updated to use session_manager
    s = domain.session_manager.get(sid)  # Updated to use session_manager
    return {"ok": True, "session_id": sid, "status": s.status, "row": row, "data": new_row}


@bg_app.post("/sessions/{sid}/accept_and_proceed")
def accept(sid: str):
    """
    Mark a session as accepted, locking it from further edits.
    Automatically saves the session data to a CSV file as part of the acceptance.

    Returns the file path where the data was saved.
    """
    file_path = domain.session_manager.accept(sid)
    s = domain.session_manager.get(sid)
    return {
        "ok": True,
        "session_id": sid,
        "status": s.status,
        "file_path": file_path,
        "message": "Session accepted and data saved successfully"
    }