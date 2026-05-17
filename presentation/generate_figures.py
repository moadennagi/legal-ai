"""Génère les figures d'évaluation pour la soutenance à partir des scores réels.

Lit uniquement les artefacts déjà produits par le pipeline d'évaluation
(`evals/summary.json`) — aucune logique RAGAS n'est réimplémentée ici, c'est
purement de la dataviz de présentation.

Usage:
    python presentation/generate_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SUMMARY = ROOT / "evals" / "summary.json"
OUT_DIRS = [Path(__file__).resolve().parent / "figures", ROOT / "evals" / "figures"]

METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
METRIC_LABELS = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer\nrelevancy",
    "context_precision": "Context\nprecision",
    "context_recall": "Context\nrecall",
}
# Palette sobre institutionnelle (bleu nuit / gris / un accent), académique
COLORS = {
    "baseline": "#9aa5b1",       # gris clair
    "+rerank": "#1b2a4a",        # bleu nuit (config retenue)
    "+hyde": "#c5ccd6",          # gris très clair
    "hyde+rerank": "#5d7299",    # bleu-gris
}
MODEL_COLORS = {
    "qwen2.5:7b": "#1b2a4a",     # bleu nuit
    "mistral:7b": "#5d7299",     # bleu-gris
    "gemma2:9b": "#9aa5b1",      # gris
}
NAVY = "#1b2a4a"
ACCENT = "#2e86c1"


def load_canonical_runs() -> list[dict]:
    """Charge les runs canoniques (ceux qui portent la clé `embedding_model`)."""
    runs = []
    with SUMMARY.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "embedding_model" not in row:  # ignorer les anciens runs incomplets
                continue
            runs.append(row)
    return runs


def config_label(row: dict) -> str:
    hyde, rerank = row.get("hyde", False), row.get("rerank", False)
    if hyde and rerank:
        return "hyde+rerank"
    if rerank:
        return "+rerank"
    if hyde:
        return "+hyde"
    return "baseline"


def runs_for_model(runs: list[dict], model: str) -> dict[str, dict]:
    """Retourne {config_label: scores} pour un modèle de génération donné."""
    out = {}
    for row in runs:
        if row.get("generation_model") == model:
            out[config_label(row)] = row["scores"]
    return out


def savefig(fig, name: str) -> None:
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_ablation(runs: list[dict], model: str = "qwen2.5:7b") -> None:
    """Barres groupées des 4 métriques × 4 configs pour le modèle par défaut."""
    data = runs_for_model(runs, model)
    configs = ["baseline", "+rerank", "+hyde", "hyde+rerank"]
    configs = [c for c in configs if c in data]

    x = np.arange(len(METRICS))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for i, cfg in enumerate(configs):
        scores = [data[cfg][m] for m in METRICS]
        bars = ax.bar(x + (i - (len(configs) - 1) / 2) * width, scores, width,
                      label=cfg, color=COLORS.get(cfg, "#888"))
        for b, s in zip(bars, scores):
            ax.text(b.get_x() + b.get_width() / 2, s + 0.01, f"{s:.2f}",
                    ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score RAGAS")
    ax.set_title(f"Ablation HyDE / Reranking — {model}", fontweight="bold")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.08), frameon=False)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    savefig(fig, "fig_ablation.png")


def fig_reranking(runs: list[dict], model: str = "qwen2.5:7b") -> None:
    """Focus : effet du reranking sur context_precision (l'effet le plus net)."""
    data = runs_for_model(runs, model)
    pairs = [("baseline", "+rerank"), ("+hyde", "hyde+rerank")]
    labels = ["Sans HyDE", "Avec HyDE"]
    before = [data[a]["context_precision"] for a, _ in pairs]
    after = [data[b]["context_precision"] for _, b in pairs]

    x = np.arange(len(pairs))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7.5, 5))
    b1 = ax.bar(x - width / 2, before, width, label="sans rerank", color="#7f8c9b")
    b2 = ax.bar(x + width / 2, after, width, label="avec rerank", color="#2e86c1")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                    f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Context precision")
    ax.set_title(f"Le reranking améliore la précision du contexte — {model}",
                 fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    savefig(fig, "fig_reranking.png")


def fig_models(runs: list[dict], config: str = "hyde+rerank") -> None:
    """Comparaison des 3 modèles de génération sur la meilleure configuration."""
    models = ["qwen2.5:7b", "mistral:7b", "gemma2:9b"]
    models = [m for m in models if config in runs_for_model(runs, m)]

    x = np.arange(len(METRICS))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for i, model in enumerate(models):
        scores = [runs_for_model(runs, model)[config][m] for m in METRICS]
        bars = ax.bar(x + (i - (len(models) - 1) / 2) * width, scores, width,
                      label=model, color=MODEL_COLORS.get(model, "#888"))
        for b, s in zip(bars, scores):
            ax.text(b.get_x() + b.get_width() / 2, s + 0.01, f"{s:.2f}",
                    ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score RAGAS")
    ax.set_title(f"Comparaison des modèles de génération (config {config})",
                 fontweight="bold")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.08), frameon=False)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    savefig(fig, "fig_models.png")


def fig_heatmap(runs: list[dict], model: str = "qwen2.5:7b") -> None:
    """Heatmap config × métrique pour une vue d'ensemble."""
    data = runs_for_model(runs, model)
    configs = [c for c in ["baseline", "+rerank", "+hyde", "hyde+rerank"] if c in data]
    matrix = np.array([[data[c][m] for m in METRICS] for c in configs])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(matrix, cmap="YlGnBu", vmin=0.6, vmax=0.95, aspect="auto")
    ax.set_xticks(range(len(METRICS)))
    ax.set_xticklabels([METRIC_LABELS[m].replace("\n", " ") for m in METRICS])
    ax.set_yticks(range(len(configs)))
    ax.set_yticklabels(configs)
    for i in range(len(configs)):
        for j in range(len(METRICS)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    color="black", fontsize=10)
    ax.set_title(f"Vue d'ensemble des scores — {model}", fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Score")
    savefig(fig, "fig_heatmap.png")


def fig_precision_config(runs: list[dict]) -> None:
    """Context precision moyenne par config, agrégée sur les 3 modèles.

    Matérialise l'argument central de H1 : le reranking maximise la précision du
    contexte (+5 à 8 pts) tandis que HyDE la dégrade → config retenue = « +rerank ».
    """
    configs = ["baseline", "+rerank", "+hyde", "hyde+rerank"]
    models = sorted({r["generation_model"] for r in runs})
    means = {}
    for cfg in configs:
        per_model = [
            runs_for_model(runs, m)[cfg]["context_precision"]
            for m in models if cfg in runs_for_model(runs, m)
        ]
        if per_model:
            means[cfg] = float(np.mean(per_model))
    configs = [c for c in configs if c in means]

    fig, ax = plt.subplots(figsize=(8, 5))
    values = [means[c] for c in configs]
    best = max(means, key=means.get)
    colors = [ACCENT if c == best else "#9aa5b1" for c in configs]
    bars = ax.bar(configs, values, color=colors, width=0.6)
    for b, cfg, v in zip(bars, configs, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.006, f"{v:.3f}",
                ha="center", va="bottom", fontsize=11,
                fontweight="bold" if cfg == best else "normal")
    ax.set_ylim(0.65, 0.90)
    ax.set_ylabel("Context precision moyenne\n(3 modèles de génération)")
    ax.set_title("Le reranking maximise la précision ; HyDE la dégrade",
                 fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    savefig(fig, "fig_precision_config.png")


def fig_scores_globaux(runs: list[dict]) -> None:
    """Score moyen (moyenne des 4 métriques) par configuration, groupé par modèle.

    Vue d'ensemble des 12 configurations pour la synthèse d'ablation (3.2.3).
    Montre honnêtement que hyde+rerank ≈ +rerank ; le choix de +rerank se justifie
    par la stabilité, le coût (1 seule passe d'embedding) et la précision.
    """
    configs = ["baseline", "+rerank", "+hyde", "hyde+rerank"]
    models = ["qwen2.5:7b", "mistral:7b", "gemma2:9b"]
    models = [m for m in models if runs_for_model(runs, m)]

    x = np.arange(len(configs))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for i, model in enumerate(models):
        data = runs_for_model(runs, model)
        scores = [float(np.mean([data[c][m] for m in METRICS])) if c in data else 0
                  for c in configs]
        bars = ax.bar(x + (i - (len(models) - 1) / 2) * width, scores, width,
                      label=model, color=MODEL_COLORS.get(model, "#888"))
        for b, s in zip(bars, scores):
            ax.text(b.get_x() + b.get_width() / 2, s + 0.004, f"{s:.3f}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_ylim(0.70, 0.82)
    ax.set_ylabel("Score moyen RAGAS (moyenne des 4 métriques)")
    ax.set_title("Score global par configuration et par modèle (12 configurations)",
                 fontweight="bold")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.08), frameon=False)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    savefig(fig, "fig_scores_globaux.png")


def main() -> None:
    runs = load_canonical_runs()
    if not runs:
        raise SystemExit(f"Aucun run canonique trouvé dans {SUMMARY}")
    print(f"{len(runs)} runs canoniques chargés "
          f"({len({r['generation_model'] for r in runs})} modèles)")
    fig_ablation(runs)
    fig_reranking(runs)
    fig_models(runs)
    fig_heatmap(runs)
    fig_precision_config(runs)
    fig_scores_globaux(runs)
    print("Figures écrites dans :", ", ".join(str(d) for d in OUT_DIRS))


if __name__ == "__main__":
    main()
