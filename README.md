# Football Rules & Tactics Assistant

## Overview

A graduation-project document assistant that answers football rules and tactics questions using retrieval-augmented generation (RAG). The notebook indexes 12 football documents in ChromaDB. The FastAPI backend retrieves relevant passages and asks a local Ollama model to answer using that context. The Streamlit frontend displays answers and source filenames.

## Tech stack

- Python 3.12
- Jupyter, pandas and NumPy for the RAG notebook
- sentence-transformers for embeddings
- ChromaDB for persistent vector search
- Ollama with `phi3:mini` for answer generation
- FastAPI, Pydantic and Uvicorn for the backend
- Streamlit and requests for the frontend
- pytest for backend tests

## Project structure

```text
ITI-Level-2-Final-project/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── api/routes/query.py
│   │   ├── core/config.py
│   │   ├── schemas/query.py
│   │   └── services/
│   │       ├── retrieval.py
│   │       └── generation.py
│   ├── tests/test_query.py
│   ├── requirements.txt
│   ├── .env.example
│   └── Dockerfile
├── frontend/
│   ├── app.py
│   ├── api_client.py
│   ├── requirements.txt
│   └── .env.example
├── notebooks/rag_pipeline.ipynb
├── data/
│   ├── raw_docs/
│   └── vector_store/
├── .gitignore
└── README.md
```

## Setup

From the project root, create and activate a virtual environment. These commands use Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend/requirements.txt -r frontend/requirements.txt
python -m pip install python-dotenv
```

Install and start Ollama, then download the model:

```powershell
ollama pull phi3:mini
```

The backend needs the exported Chroma collection and `config.json` in `data/vector_store/`. If they are missing, install Jupyter and pandas and run `notebooks/rag_pipeline.ipynb` from top to bottom first.

## Run the backend

In an activated terminal, starting at the project root:

```powershell
cd backend
uvicorn app.main:app --reload
```

The backend listens on port 8000. Open `http://localhost:8000/docs` for the interactive API documentation. `GET /health` reports whether startup completed.

## Run the frontend

In a second terminal, starting at the project root:

```powershell
.\.venv\Scripts\Activate.ps1
cd frontend
Copy-Item .env.example .env
streamlit run app.py
```

Set `API_BASE_URL=http://localhost:8000` in `frontend/.env`. Keep the backend and Ollama running. Open `http://localhost:8501` to ask football questions and view answers with sources. Local `.env` files are excluded from Git.

## API example

Send a JSON question to `POST /query` (PowerShell):

```powershell
curl.exe -X POST http://localhost:8000/query -H "Content-Type: application/json" --data-raw '{"question":"What is the offside trap?"}'
```

The response contains `answer` (a string) and `sources` (a list of source filenames). Missing or blank questions return HTTP 422.

Screenshots and evaluation results below
