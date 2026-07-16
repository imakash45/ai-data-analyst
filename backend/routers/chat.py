"""
/chat - natural language Q&A over the uploaded dataset. Grounded on
        server-computed stats (and /train results if available), never
        on the raw dataframe directly.
"""
from __future__ import annotations
from fastapi import APIRouter, HTTPException

from core.schemas import ChatRequest, ChatResponse
from core.session_store import store
from core.chat import build_dataset_context, generate_chat_answer

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat_with_dataset(request: ChatRequest):
    session = store.get(request.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found. Upload a file first.")

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    context = build_dataset_context(session.df, session)
    answer, generated = generate_chat_answer(request.question, context)

    return ChatResponse(
        session_id=request.session_id,
        question=request.question,
        answer=answer,
        answer_generated=generated,
    )