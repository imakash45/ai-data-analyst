"""
/report - returns a downloadable PDF, not JSON. Uses StreamingResponse
          with a Content-Disposition header so browsers/Streamlit trigger
          a file download instead of trying to render it inline.
"""
from __future__ import annotations
import io
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from core.schemas import ReportRequest
from core.session_store import store
from core.report import generate_report_pdf

router = APIRouter(tags=["report"])


@router.post("/report")
async def generate_report(request: ReportRequest):
    session = store.get(request.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found. Upload a file first.")

    pdf_bytes = generate_report_pdf(session)

    filename = f"report_{request.session_id[:8]}.pdf"
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )