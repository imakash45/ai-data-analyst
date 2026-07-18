# 📊 AI Data Analyst

An end-to-end data analysis platform that takes any uploaded CSV or Excel dataset and automatically produces exploratory data analysis, trains and compares multiple ML models, explains predictions with SHAP, and answers natural-language questions about the data — all through a REST API with a Streamlit frontend.

**🔗 Live demo:** [ai-analyst-data.streamlit.app](https://ai-analyst-data.streamlit.app)

> Note: the backend runs on Render's free tier, which sleeps after inactivity. The first request after idle time may take 30-60 seconds to wake up — please wait a moment rather than assuming it's broken.

---

## What it does

1. **Upload** — drop in a CSV or Excel file; get an instant profile (row/column counts, inferred types, missing values, duplicates)
2. **Clean** — choose per-column imputation strategies (mean/median/mode/drop), remove duplicates, and manually override any column's inferred type
3. **Explore (EDA)** — auto-generated statistics, correlation heatmap, distributions, category breakdowns, and an AI-written insight paragraph
4. **Train** — pick a target column; task type (regression/classification) is auto-detected. Trains Linear/Logistic Regression, Random Forest, and XGBoost, and compares them on a shared metric table. Optionally exclude specific columns from training.
5. **Explain** — SHAP feature importance for the best-performing model
6. **Chat** — ask natural-language questions about the dataset, grounded entirely in the stats already computed (never hallucinated from the raw data)
7. **Report** — download a single PDF combining the overview, insight, charts, and model results

---

## Why this exists

Most "auto-EDA" tools stop at descriptive statistics. This project pairs that with a full automated ML comparison layer, model explainability, and a chat interface grounded in real computed numbers — while staying transparent about every automatic decision (nothing is silently dropped or transformed without being shown to the user).

## Design decisions worth knowing

- **Deterministic feature encoding, not black-box AutoML.** Every column's treatment (one-hot, label-encode, drop, datetime feature extraction) follows a fixed, inspectable rule based on inferred type and cardinality — visible to the user via a "dropped columns" list on every training run, not hidden.
- **SHAP with `tree_path_dependent` perturbation.** Chosen specifically because newer XGBoost versions mark categorical-like integer columns internally in a way that SHAP's default "interventional" mode doesn't support — `tree_path_dependent` avoids that entirely and needs no background dataset.
- **Chat is grounded, not free-form.** The LLM (Groq/Llama 3.3 70B) never sees the raw dataframe — only precomputed JSON statistics — so answers can't hallucinate numbers that aren't actually in the data.
- **Manual column exclusion + type override.** Auto-inference gets misclassifications wrong sometimes (e.g. a meaningful column mistaken for free text); rather than silently living with that, the user can override a column's type or exclude it from training entirely, with the reasoning always shown.

---

## Tech stack

| Layer | Tools |
|---|---|
| Backend | FastAPI, Pydantic |
| Data processing | Pandas, NumPy |
| ML | Scikit-learn (Linear/Logistic Regression, Random Forest), XGBoost |
| Explainability | SHAP |
| AI layer | Groq API (Llama 3.3 70B) |
| Charts | Plotly, Matplotlib |
| Report generation | fpdf2 |
| Frontend | Streamlit |
| Deployment | Render (backend), Streamlit Cloud (frontend) |

## Architecture

```
ai-data-analyst/
├── backend/
│   ├── main.py                 # FastAPI app entrypoint
│   ├── routers/
│   │   ├── ingestion.py        # /upload, /clean
│   │   ├── eda.py              # /eda
│   │   ├── ml.py                # /train
│   │   ├── explain.py           # /explain (SHAP)
│   │   ├── chat.py               # /chat (Groq)
│   │   └── report.py              # /report (PDF export)
│   ├── core/
│   │   ├── cleaning.py         # type inference, imputation, outlier detection
│   │   ├── eda.py               # stats computation
│   │   ├── ml.py                 # feature encoding, model training/comparison
│   │   ├── explain.py             # SHAP feature importance
│   │   ├── insight.py              # AI EDA insight generation
│   │   ├── chat.py                  # grounded chat context builder
│   │   ├── report.py                 # PDF assembly
│   │   ├── schemas.py                 # Pydantic request/response models
│   │   └── session_store.py            # in-memory session state
│   └── requirements.txt
├── frontend/
│   ├── app.py                  # Streamlit UI, 7-step workflow
│   ├── api_client.py            # thin wrapper over backend REST calls
│   ├── styles.py                  # custom CSS for the UI
│   └── requirements.txt
└── datasets/                   # sample datasets for demoing
```

Two independently deployed services: the Streamlit frontend calls the FastAPI backend entirely over REST — no shared process or database, just HTTP.

---

## Running locally

**Backend:**
```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
# create a .env file with: GROQ_API_KEY=your_key_here
uvicorn main:app --reload --port 8000
```

**Frontend** (in a separate terminal):
```bash
cd frontend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

By default, `frontend/api_client.py` points at the deployed backend. To run fully locally, change `BACKEND_URL` in that file to `http://127.0.0.1:8000`.

---

## API endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/upload` | POST | Upload a CSV/Excel file, get a column profile |
| `/clean` | POST | Apply imputation rules, drop duplicates, override column types |
| `/eda` | POST | Compute statistics, correlations, distributions, AI insight |
| `/train` | POST | Train and compare Linear/Logistic Regression, Random Forest, XGBoost |
| `/explain` | POST | SHAP feature importance for the best model |
| `/chat` | POST | Ask a natural-language question grounded in computed stats |
| `/report` | POST | Generate a downloadable PDF report |

Full interactive docs (Swagger UI) available at `/docs` on the deployed backend.

---

## Known limitations

- Free-text columns (long-form text) are excluded from training — no NLP/embedding support
- No hyperparameter tuning — models train with fixed, reasonable defaults for a fair comparison
- Free-tier hosting means the backend cold-starts after ~15 minutes of inactivity
- Session state is in-memory — restarting the backend clears any active sessions

---

## Author

**Akash Kumar Pandit**
[GitHub](https://github.com/imakash45) · [LinkedIn](https://linkedin.com/in/imakash45)