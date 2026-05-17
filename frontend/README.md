---
title: Legal AI Demo
emoji: ⚖️
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.39.0
app_file: streamlit_app.py
pinned: false
license: mit
---

# Legal AI — Streamlit demo

Public demo of the [Legal AI](https://github.com/moadennagi/legal-ai) RAG system, hosted on Hugging Face Spaces.

## Local development

```bash
cd frontend
pip install -r requirements.txt
API_URL=http://localhost:8000 streamlit run streamlit_app.py
```

## Deployment to Hugging Face Spaces

1. Create a new Space at <https://huggingface.co/new-space> (type: **Streamlit**)
2. Push the contents of this directory to the Space repository:
   ```bash
   cd frontend
   git init
   git remote add space https://huggingface.co/spaces/<your-username>/legal-ai-demo
   git add .
   git commit -m "Initial Streamlit app"
   git push space main
   ```
3. Set the **secret** `API_URL` in the Space settings, pointing to your deployed FastAPI backend (e.g. `https://your-api.fly.dev`).

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `API_URL` | `http://localhost:8000` | URL of the Legal AI FastAPI backend |
| `API_TIMEOUT` | `60` | HTTP timeout (seconds) for backend calls |
