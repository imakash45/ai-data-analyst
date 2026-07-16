"""
AI Data Analyst — FastAPI backend entrypoint.

Run locally:
    cd backend
    uvicorn main:app --reload --port 8000

Endpoints are added router-by-router as each module is built:
    /upload, /clean   -> routers/ingestion.py   (Module 1 — this one)
    /eda              -> routers/eda.py          (Module 2)
    /train            -> routers/ml.py           (Module 3)
    /explain           -> routers/explain.py       (Module 4)
    /chat              -> routers/chat.py           (Module 5)
    /report            -> routers/report.py          (Module 6)
"""

from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import ingestion, eda, ml, explain, chat, report

app = FastAPI(
    title="AI Data Analyst API",
    description="Upload any dataset and get auto-EDA, auto-ML, and an AI analyst chat layer.",
    version="0.1.0",
)

# Streamlit frontend (local + Streamlit Cloud) will call this API cross-origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to the deployed Streamlit URL once known
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingestion.router)
app.include_router(eda.router)
app.include_router(ml.router)
app.include_router(explain.router)
app.include_router(chat.router)
app.include_router(report.router)

@app.get("/")
def health_check():
    return {"status": "ok", "service": "ai-data-analyst-backend"}
