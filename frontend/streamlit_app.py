"""Streamlit demo UI for the Legal AI RAG system.

Three tabs:
  - Démo : interactive Q&A against the FastAPI backend
  - Résultats RAGAS : visualisation of the evaluation runs from evals/summary.json
  - Méthodologie : architecture + evaluation methodology summary

The backend URL is configurable through the `API_URL` env var.
Deployed as a single HuggingFace Space (Docker) — see README_HFSPACE.md.
"""

import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://localhost:8000")
API_TIMEOUT = int(os.getenv("API_TIMEOUT", "60"))

# Path to evaluation summary — tried in order; first existing wins.
SUMMARY_CANDIDATES = [
    Path(__file__).parent.parent / "evals" / "summary.json",
    Path("/app/evals/summary.json"),
    Path("evals/summary.json"),
]

SAMPLE_QUESTIONS = [
    "Quelles sont les conditions d'octroi d'une aide aux exploitations agricoles ?",
    "Quels sont les critères d'éligibilité pour obtenir un agrément forestier ?",
    "Comment est calculée la taxe sur la valeur ajoutée pour les sociétés ?",
    "Quelles sont les obligations d'un administrateur de société anonyme ?",
    "Quel est le régime fiscal applicable aux OPCVM ?",
    "Quelles sont les démarches pour créer une compagnie d'assurance ?",
    "Quels documents sont nécessaires pour l'importation de produits agricoles ?",
    "Quelles sont les sanctions prévues en cas de pratique anticoncurrentielle ?",
    "Comment se déroule la procédure de constitution d'une coopérative ?",
    "Quels sont les droits et obligations des salariés en CDD ?",
]

METRIC_LABELS = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer relevancy",
    "context_precision": "Context precision",
    "context_recall": "Context recall",
}


# ─── Page configuration ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="Legal AI — RAG démo",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚖️ Legal AI")
    st.markdown(
        "Système RAG souverain pour textes juridiques en français, "
        "validé sur le **Bulletin Officiel du Royaume du Maroc**."
    )

    st.divider()
    st.subheader("À propos")
    st.markdown(
        """
        - 📚 **Corpus** : Bulletin Officiel marocain (extrait)
        - 🧠 **Embedding** : `bge-m3` (multilingue)
        - 🔍 **Retrieval** : pgvector + cross-encoder reranking
        - 🤖 **Génération** : Mistral 7B via Together AI
        - 📊 **Évaluation** : RAGAS (4 métriques)
        """
    )

    st.divider()
    st.subheader("Liens")
    st.markdown(
        """
        - [📦 Code source (GitHub)](https://github.com/moadennagi/legal-ai)
        - [📖 Documentation](https://github.com/moadennagi/legal-ai#readme)
        - [📊 Méthodologie d'évaluation](https://github.com/moadennagi/legal-ai/blob/main/docs/EVALUATION.md)
        """
    )

    st.divider()
    api_status = st.empty()


# ─── Backend health check ────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def check_api_health() -> dict[str, Any] | None:
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


health = check_api_health()
if health:
    api_status.success(f"🟢 API en ligne — `{health.get('chat_model', 'unknown')}`")
else:
    api_status.error("🔴 API hors ligne. Vérifiez `API_URL`.")


# ─── Helpers ─────────────────────────────────────────────────────────────────

def call_rag_api(query: str) -> dict[str, Any]:
    """Call the OpenAI-compatible /v1/chat/completions endpoint."""
    payload = {
        "model": "legal-ai-rag",
        "messages": [{"role": "user", "content": query}],
    }
    response = requests.post(
        f"{API_URL}/v1/chat/completions",
        json=payload,
        timeout=API_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=3600)
def load_ragas_runs() -> pd.DataFrame:
    """Load and flatten RAGAS runs from summary.json.

    Keeps only entries that have both `embedding_model` and `generation_model`
    fields (the most recent, fully-parameterised runs)."""
    path = next((p for p in SUMMARY_CANDIDATES if p.exists()), None)
    if path is None:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if "embedding_model" not in entry or "generation_model" not in entry:
                continue
            flat = {
                "embedding": entry["embedding_model"],
                "generation": entry["generation_model"],
                "hyde": bool(entry.get("hyde", False)),
                "rerank": bool(entry.get("rerank", False)),
                **{k: entry["scores"].get(k) for k in METRIC_LABELS},
            }
            flat["mean"] = sum(flat[k] for k in METRIC_LABELS) / len(METRIC_LABELS)
            rows.append(flat)
    return pd.DataFrame(rows)


# ─── Tabs ────────────────────────────────────────────────────────────────────

tab_demo, tab_ragas, tab_method = st.tabs(
    ["💬 Démo", "📊 Résultats RAGAS", "ℹ️ Méthodologie"]
)


# ─── Tab 1 : Démo ────────────────────────────────────────────────────────────

with tab_demo:
    st.title("Démo Legal AI — RAG juridique")

    st.markdown(
        "Posez une question sur un texte juridique du Bulletin Officiel marocain. "
        "Le système recherche les passages pertinents dans la base, les classe par pertinence, "
        "puis génère une réponse ancrée dans les sources retrouvées."
    )

    st.warning(
        "⚠️ **Démo académique** — Les réponses générées peuvent contenir des erreurs "
        "et **ne constituent pas un conseil juridique**. Vérifiez toujours les sources citées.",
        icon="⚠️",
    )

    st.subheader("Posez votre question")

    col1, col2 = st.columns([3, 1])
    with col1:
        selected_sample = st.selectbox(
            "Exemples de questions",
            options=["— sélectionner un exemple —"] + SAMPLE_QUESTIONS,
            index=0,
        )

    if "current_question" not in st.session_state:
        st.session_state.current_question = ""

    if selected_sample != "— sélectionner un exemple —":
        st.session_state.current_question = selected_sample

    question = st.text_area(
        "Votre question",
        value=st.session_state.current_question,
        height=100,
        placeholder="Ex: Quelles sont les conditions d'octroi d'une aide aux exploitations agricoles ?",
    )

    submit = st.button(
        "🔍 Demander",
        type="primary",
        use_container_width=True,
        disabled=(health is None),
    )

    if submit and question.strip():
        with st.spinner("🔎 Recherche et génération en cours… (10-30 sec)"):
            start = time.time()
            try:
                result = call_rag_api(question.strip())
                elapsed = time.time() - start

                answer = result["choices"][0]["message"]["content"]

                st.subheader("Réponse")
                st.markdown(answer)

                st.caption(
                    f"⏱️ Réponse générée en {elapsed:.1f}s · "
                    f"Modèle : `{result.get('model', 'unknown')}`"
                )

                with st.expander("📚 Sources consultées"):
                    st.info(
                        "La version actuelle de l'API ne renvoie pas encore le détail "
                        "des passages utilisés. Les sources sont citées dans la réponse "
                        "elle-même lorsque le modèle les mentionne."
                    )

            except requests.exceptions.Timeout:
                st.error(
                    f"⏰ Délai dépassé (>{API_TIMEOUT}s). "
                    "Le modèle est peut-être en cours de chargement."
                )
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    st.error("⚠️ Trop de requêtes. Limite : 10/min par IP. Réessayez dans une minute.")
                else:
                    st.error(f"❌ Erreur API : {e.response.status_code} — {e.response.text}")
            except Exception as e:
                st.error(f"❌ Erreur inattendue : {e}")

    elif submit and not question.strip():
        st.warning("Veuillez saisir une question avant de soumettre.")


# ─── Tab 2 : Résultats RAGAS ─────────────────────────────────────────────────

with tab_ragas:
    st.title("📊 Résultats de l'évaluation RAGAS")

    st.markdown(
        "Évaluation comparative de la pipeline RAG sur un jeu de Q/R synthétique "
        "généré par un LLM différent du LLM de génération (cf. méthodologie §3.6 du mémoire). "
        "Quatre métriques RAGAS sont rapportées :"
    )
    st.markdown(
        """
        - **Faithfulness** : la réponse est-elle factuellement supportée par les chunks récupérés ?
        - **Answer relevancy** : la réponse répond-elle bien à la question ?
        - **Context precision** : les chunks récupérés sont-ils pertinents ?
        - **Context recall** : les chunks couvrent-ils toute l'information nécessaire ?
        """
    )

    df = load_ragas_runs()
    if df.empty:
        st.warning("Aucun résultat RAGAS trouvé. Lance d'abord `python -m evals` pour produire `evals/summary.json`.")
    else:
        # ─── Best config highlight ──────────────────────────────────────────
        best = df.loc[df["mean"].idxmax()]
        st.subheader("🏆 Meilleure configuration")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Modèle de génération", best["generation"])
        col2.metric("HyDE", "✅" if best["hyde"] else "❌")
        col3.metric("Reranking", "✅" if best["rerank"] else "❌")
        col4.metric("Score moyen", f"{best['mean']:.3f}")

        cols = st.columns(4)
        for i, (key, label) in enumerate(METRIC_LABELS.items()):
            cols[i].metric(label, f"{best[key]:.3f}")

        st.divider()

        # ─── Filter controls ────────────────────────────────────────────────
        st.subheader("Explorer toutes les configurations")
        f1, f2, f3 = st.columns(3)
        gen_filter = f1.multiselect(
            "Modèle de génération",
            sorted(df["generation"].unique()),
            default=sorted(df["generation"].unique()),
        )
        hyde_filter = f2.selectbox("HyDE", ["Tous", "Activé", "Désactivé"])
        rerank_filter = f3.selectbox("Reranking", ["Tous", "Activé", "Désactivé"])

        filtered = df[df["generation"].isin(gen_filter)]
        if hyde_filter != "Tous":
            filtered = filtered[filtered["hyde"] == (hyde_filter == "Activé")]
        if rerank_filter != "Tous":
            filtered = filtered[filtered["rerank"] == (rerank_filter == "Activé")]

        # ─── Table ──────────────────────────────────────────────────────────
        display = filtered.copy()
        display["hyde"] = display["hyde"].map({True: "✅", False: "❌"})
        display["rerank"] = display["rerank"].map({True: "✅", False: "❌"})
        display = display.rename(columns={
            "generation": "Génération",
            "embedding": "Embedding",
            "hyde": "HyDE",
            "rerank": "Rerank",
            "faithfulness": "Faithf.",
            "answer_relevancy": "Ans. rel.",
            "context_precision": "Ctx. prec.",
            "context_recall": "Ctx. rec.",
            "mean": "Moyenne",
        })
        st.dataframe(
            display.style.format({
                "Faithf.": "{:.3f}",
                "Ans. rel.": "{:.3f}",
                "Ctx. prec.": "{:.3f}",
                "Ctx. rec.": "{:.3f}",
                "Moyenne": "{:.3f}",
            }).background_gradient(
                subset=["Faithf.", "Ans. rel.", "Ctx. prec.", "Ctx. rec.", "Moyenne"],
                cmap="RdYlGn",
                vmin=0.5,
                vmax=1.0,
            ),
            use_container_width=True,
            hide_index=True,
        )

        # ─── Bar chart : config vs metric ───────────────────────────────────
        st.subheader("Comparaison visuelle")
        chart_df = filtered.copy()
        chart_df["config"] = (
            chart_df["generation"]
            + " | "
            + chart_df["hyde"].map({True: "HyDE+", False: "HyDE−"})
            + " "
            + chart_df["rerank"].map({True: "RR+", False: "RR−"})
        )
        chart_data = chart_df.set_index("config")[list(METRIC_LABELS.keys())]
        st.bar_chart(chart_data, height=400)

        st.caption(
            f"Total runs affichés : {len(filtered)} · "
            f"Juge LLM : `gpt-4o-mini` · "
            f"Embedding : `bge-m3`"
        )


# ─── Tab 3 : Méthodologie ────────────────────────────────────────────────────

with tab_method:
    st.title("ℹ️ Méthodologie et architecture")

    st.subheader("Pipeline en quatre phases")
    st.markdown(
        """
        1. **Ingestion** : crawling du site `sgg.gov.ma`, téléchargement concurrent des PDFs
        2. **Extraction** : Docling convertit chaque PDF en Markdown (OCR désactivé)
        3. **Embedding** : un splitter spécifique au Bulletin Officiel normalise la hiérarchie
           des titres juridiques (DAHIR, Loi, Décret, Chapitre, Section…), puis chunke le
           Markdown avec préservation des en-têtes. Embedding via `bge-m3` (1024 dim).
        4. **RAG** : HyDE (embedding de la question + d'une réponse hypothétique) +
           recherche pgvector (IVFFLAT, 10 probes) + reranking cross-encoder
           (`ms-marco-MiniLM-L-6-v2`).
        """
    )

    st.subheader("Architecture de déploiement (démo)")
    st.code(
        """┌─────────── HuggingFace Space (Docker, CPU) ────────────┐
│                                                         │
│  Streamlit (port 7860)                                  │
│     ↓                                                   │
│  FastAPI (port 8000, OpenAI-compatible)                 │
│     ├─ embeddings → Ollama bge-m3 (CPU, local)          │
│     ├─ retrieval  → Postgres + pgvector (local)         │
│     └─ génération → Together AI (Mistral 7B)            │
└─────────────────────────────────────────────────────────┘""",
        language="text",
    )

    st.subheader("Méthodologie d'évaluation RAGAS")
    st.markdown(
        """
        - **Jeu de Q/R** : généré par un LLM **différent** du LLM de génération évalué,
          pour éviter le biais d'évaluation circulaire. Chaque paire est ensuite
          revue manuellement avant inclusion dans le dataset final.
        - **Juge** : `gpt-4o-mini` (provider externe), invariant pour toutes les configs.
        - **Variables comparées** : modèle de génération (qwen2.5:7b, mistral:7b, gemma2:9b),
          HyDE on/off, reranking on/off.
        - **Métriques** : faithfulness, answer relevancy, context precision, context recall.

        Voir le mémoire §3.6 et [`docs/EVALUATION.md`](https://github.com/moadennagi/legal-ai/blob/main/docs/EVALUATION.md).
        """
    )

    st.subheader("Limitations connues")
    st.markdown(
        """
        - **Latence** : sur CPU (démo gratuite), une requête prend 10-30 sec
          (embedding bge-m3 ~2 sec + génération externe + reranking).
        - **Corpus** : la démo expose un sous-ensemble du Bulletin Officiel (3 domaines).
          Le mémoire évalue sur un corpus plus large.
        - **Pas de citation explicite des sources** dans la réponse pour le moment
          (les chunks récupérés sont utilisés par le LLM mais pas exposés par l'API).
        """
    )


# ─── Footer ──────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "Projet de recherche académique · "
    "Code source [MIT License](https://github.com/moadennagi/legal-ai/blob/main/LICENSE) · "
    "© 2026 Moad Ennagi"
)
