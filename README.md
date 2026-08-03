# AI-Powered Smart Retail & Customer Intelligence Platform

A production-ready Python platform that gives retailers **face-based customer
recognition**, **product image classification**, **review sentiment analysis**,
an **FAQ chatbot**, and a **live business-intelligence dashboard** — served by a
FastAPI backend with a unified ML pipeline, fully containerised and CI-tested.

## Architecture

```
┌────────────┐     ┌──────────────────────────────────────────────┐     ┌────────────┐
│  Streamlit │ ──▶ │  FastAPI  (app/)                              │ ──▶ │  SQLite /  │
│  Dashboard │ ◀── │   routers → services → ML pipeline → models   │     │ PostgreSQL │
└────────────┘     │   API-key auth · CORS · logging · exceptions  │     └────────────┘
                   └──────────────────────────────────────────────┘
```

| Component | Technology | Notes |
|---|---|---|
| Face recognition | `face_recognition` (dlib) | 128-d encodings stored per customer; OpenCV Haar fallback |
| Product classification | MobileNetV2 (transfer learning) | fine-tuned on Fashion-MNIST; ImageNet fallback |
| Sentiment analysis | TF-IDF + Logistic Regression | lexicon fallback until trained |
| FAQ chatbot | rule-based + ML intent classifier | driven by `data/intents.json` |
| Backend | FastAPI + Pydantic v2 + SQLAlchemy | PostgreSQL-ready |
| Dashboard | Streamlit + Plotly | visits, sentiment, chats, products |
| Deployment | Docker, docker-compose, GitHub Actions, Render | |

## Project layout

```
app/                  FastAPI application
  core/               config, security, logging, exceptions
  database/           SQLAlchemy models, session, base
  models/             ML wrappers (face, product, sentiment, chatbot)
  routers/            HTTP endpoints
  schemas/            Pydantic request/response models
  services/           orchestration + unified ML pipeline
  utils/              image/text helpers
  main.py             app entry point
training/             dataset downloader + training scripts
frontend/             Streamlit dashboard
tests/                PyTest suite (no heavy deps required)
data/                 intents.json + downloaded datasets
models/artifacts/     trained model files
scripts/              demo-data seeder
notebooks/            exploration notebook
```

## Quick start (local)

Dependencies are managed with **uv** (a single fast `pyproject.toml` + locked
`uv.lock`). Install uv first if you don't have it:

```bash
# Windows:  pip install uv        macOS/Linux:  curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
uv sync                          # core + dev tooling
uv sync --group ml               # + TensorFlow + dlib (heavy, optional)
# or: uv sync --all-groups

copy .env.example .env           # then set API_KEY
uv run python -m scripts.seed_demo_data     # optional demo records

# Terminal 1 — API
uv run uvicorn app.main:app --reload
# Swagger UI → http://localhost:8000/docs

# Terminal 2 — Dashboard
uv run streamlit run frontend/dashboard.py
# → http://localhost:8501
```

The API runs immediately with **fallback models** (lexicon sentiment, rule
chatbot, ImageNet product classifier, OpenCV face detection). Run the training
scripts below to upgrade to the full ML models.

## API endpoints

| Method | Path | Auth | Description |
|---|---|---|---|
| POST | `/recognize-face` | ✔ | Detect/recognise a customer from base64 image |
| POST | `/classify-product` | ✔ | Classify product image, returns top-k |
| POST | `/analyze-sentiment` | ✔ | Positive / Neutral / Negative |
| POST | `/chatbot` | ✔ | FAQ reply + intent |
| GET | `/dashboard/stats` | ✔ | Aggregated BI statistics |
| GET | `/health` | — | Liveness probe |

Send requests with header `X-API-Key: <your key>` (default in `.env`).

```bash
curl -X POST http://localhost:8000/analyze-sentiment \
  -H "Content-Type: application/json" -H "X-API-Key: change-me-in-production" \
  -d '{"text":"I love this store!"}'
```

## Training

```bash
uv run python -m training.download_datasets --all      # Fashion-MNIST, reviews CSV, LFW
uv run python -m training.train_sentiment --csv data/ecommerce_reviews.csv
uv run python -m training.train_product --epochs 6
uv run python -m training.train_face data/faces --seed # per-person subfolders
```

Artifacts are written to `models/artifacts/` and picked up automatically on the
next API restart.

## Tests

```bash
uv sync --frozen               # ensure core + dev are installed
uv run pytest -q               # or: uv run pytest --cov=app
```

The test suite does **not** require TensorFlow or dlib — services touching them
are stubbed.

## Docker

```bash
docker compose up --build
# API:      http://localhost:8000
# Dashboard:http://localhost:8501
```

## Deploy

- **Render:** use `render.yaml` (installs uv, `uv sync --frozen --group ml`,
  starts with `uv run`; API_KEY is auto-generated).
- **Railway:** build `uv sync --frozen --group ml`, start
  `uv run uvicorn app.main:app --host 0.0.0.0 --port $PORT`.
- **GitHub Actions:** `.github/workflows/ci.yml` uses `astral-sh/setup-uv`,
  lints (ruff) and runs the suite on every push/PR.

For PostgreSQL set `DATABASE_URL=postgresql+psycopg://user:pass@host:5432/retail`.

## Security notes

- API-key auth enforced via constant-time comparison.
- Change `API_KEY` before any non-local deployment.
- Face encodings are stored as JSON; treat them as sensitive PII.
