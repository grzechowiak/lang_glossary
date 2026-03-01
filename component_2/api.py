import datetime
import sys
import os
# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

import domain

from datetime import datetime
timestamp = datetime.now().strftime("DATE:_%Y-%m-%d_TIME:_%H-%M-%S")


class EditCellRequest(BaseModel):
    row: int = Field(..., ge=0)
    col: str
    value: str


class RegenerateRowRequest(BaseModel):
    feedback: str = Field(..., min_length=1)


app = FastAPI(title=f'Session-based Table Review API (CSV upload + Pydantic validation)_{timestamp}')


@app.post("/sessions")
async def create_session(csv: UploadFile = File(...)):
    raw = await csv.read()
    df = domain.dataframe_from_csv_bytes(raw)
    sid = domain.store.create_from_df(df)
    return {"session_id": sid, "status": "awaiting_action", "row_count": int(len(df))}


@app.get("/sessions/{sid}/show_table")
def get_table(sid: str):
    s = domain.store.get(sid)
    return {
        "session_id": sid,
        "status": s.status,
        "columns": list(s.df.columns),
        "row_count": int(len(s.df)),
        "rows": s.to_rows(),
        "change_log_tail": s.change_log[-10:],
    }


@app.patch("/sessions/{sid}/cell_modification")
def patch_cell(sid: str, req: EditCellRequest):
    domain.edit_cell(sid, req.row, req.col, req.value)
    s = domain.store.get(sid)
    return {"ok": True, "session_id": sid, "status": s.status, "rows": s.to_rows()}


@app.post("/sessions/{sid}/rows/{row}/row_regenerate")
def regen_row(sid: str, row: int, req: RegenerateRowRequest):
    new_row = domain.regenerate_row(sid, row, req.feedback.strip())
    s = domain.store.get(sid)
    return {"ok": True, "session_id": sid, "status": s.status, "row": row, "data": new_row}


@app.post("/sessions/{sid}/accept")
def accept(sid: str):
    domain.accept(sid)
    s = domain.store.get(sid)
    return {"ok": True, "session_id": sid, "status": s.status}


@app.get("/sessions/{sid}/print_my_csv", response_class=PlainTextResponse)
def csv_export(sid: str):
    s = domain.store.get(sid)
    return s.get_csv_content()

@app.post("/sessions/{sid}/save_to_csv")
def save_session_to_file(sid: str, filename_base: str = "Component_2_table"):
    """Save the session data to a CSV file in the configured output directory"""
    s = domain.store.get(sid)
    file_path = s.save_csv_to_file(filename_base)
    return {
        "ok": True,
        "session_id": sid,
        "status": s.status,
        "file_path": file_path
    }