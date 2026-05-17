# Méthodologie et résultats de l'évaluation RAGAS

> Ce document résume la méthodologie d'évaluation et les principaux résultats
> obtenus sur la pipeline RAG. Le détail complet (justification théorique,
> analyse approfondie, discussion) est traité au **chapitre 3.6 du mémoire**.

## 1. Objectif

Quantifier la qualité de la pipeline RAG selon trois axes :

1. Le modèle de génération utilisé (qwen2.5:7b, mistral:7b, gemma2:9b)
2. L'apport de **HyDE** (Hypothetical Document Embeddings)
3. L'apport du **reranking** par cross-encoder

L'ensemble des combinaisons (3 × 2 × 2 = 12 configurations) est évalué sur le
même jeu de Q/R par le même juge LLM.

## 2. Jeu de données

Le dataset d'évaluation (`evals/ragas_eval_dataset.csv`) contient **79 paires
question–réponse** couvrant trois domaines du droit marocain (social, fiscal,
sociétés) et trois profils de question (factuelle mono-article, synthèse
multi-articles, hors-corpus).

- **Génération** : produit par un LLM **différent** du LLM de génération évalué
  (cf. [`evaluation/qa_generator.py`](../src/legal_ai/evaluation/qa_generator.py))
  afin d'éviter le biais d'auto-évaluation.
- **Validation** : chaque paire est revue manuellement avant inclusion. Les
  paires écartées et leurs motifs sont documentés dans
  [`evals/README.md`](../evals/README.md).

## 3. Métriques RAGAS

Les quatre métriques retenues sont définies dans Es et al. (2023) :

| Métrique             | Cible              | Question évaluée                                                                |
|----------------------|--------------------|---------------------------------------------------------------------------------|
| **Faithfulness**     | génération         | La réponse est-elle factuellement supportée par les chunks récupérés ?          |
| **Answer relevancy** | génération         | La réponse répond-elle bien à la question posée ?                               |
| **Context precision**| retrieval          | Les chunks récupérés sont-ils pertinents (sans bruit) ?                          |
| **Context recall**   | retrieval          | Les chunks couvrent-ils toute l'information nécessaire à une réponse complète ? |

**Juge** : `gpt-4o-mini` (OpenAI), invariant pour toutes les configurations
évaluées. Le choix d'un juge **externe** au pool de modèles de génération
testés est délibéré et discuté dans le mémoire §3.6.

## 4. Résultats agrégés

Source : [`evals/summary.json`](../evals/summary.json). Embedding identique pour
toutes les lignes (`bge-m3`).

### Synthèse par modèle de génération (HyDE off, rerank off → off+rerank)

| Modèle gen.   | HyDE | Rerank | Faithf. | Ans. rel. | Ctx. prec. | Ctx. rec. |
|---------------|:----:|:------:|--------:|----------:|-----------:|----------:|
| qwen2.5:7b    |  ❌  |   ❌   | 0.706   | 0.789     | 0.804      | 0.804     |
| qwen2.5:7b    |  ❌  |   ✅   | 0.709   | 0.734     | **0.863**  | **0.823** |
| qwen2.5:7b    |  ✅  |   ❌   | 0.673   | 0.737     | 0.724      | 0.792     |
| qwen2.5:7b    |  ✅  |   ✅   | **0.738** | **0.801** | 0.836      | 0.803     |
| mistral:7b    |  ❌  |   ❌   | 0.687   | 0.849     | 0.804      | 0.807     |
| mistral:7b    |  ❌  |   ✅   | 0.641   | 0.835     | 0.852      | 0.831     |
| mistral:7b    |  ✅  |   ❌   | 0.617   | 0.769     | 0.688      | 0.814     |
| mistral:7b    |  ✅  |   ✅   | 0.631   | **0.856** | 0.849      | 0.817     |
| gemma2:9b     |  ❌  |   ❌   | 0.673   | 0.704     | 0.822      | 0.813     |
| gemma2:9b     |  ❌  |   ✅   | 0.622   | 0.715     | **0.878**  | 0.814     |
| gemma2:9b     |  ✅  |   ❌   | 0.714   | 0.655     | 0.700      | 0.796     |
| gemma2:9b     |  ✅  |   ✅   | 0.651   | 0.709     | 0.839      | 0.803     |

## 5. Conclusions principales

### 5.1 — Le reranking améliore systématiquement la précision contextuelle

Quel que soit le modèle, le rerank fait **gagner 4 à 9 points** de
`context_precision`. C'est le levier le plus régulier et le moins coûteux à
intégrer.

### 5.2 — HyDE est contre-productif sur du texte normatif

Activer HyDE **dégrade** systématiquement `context_precision` (sauf si
recompensé par le rerank). Hypothèse retenue :

> Les modèles 7B généralistes paraphrasent en français courant les questions
> de droit, là où le corpus utilise une forme normative très spécifique
> (« DAHIR portant promulgation de la loi… »). L'embedding de la réponse
> hypothétique s'écarte alors du registre du corpus et bruite la recherche.

Ce résultat contre-intuitif est l'une des contributions de l'évaluation.

### 5.3 — Le choix du modèle de génération a un effet modéré

Les écarts entre `qwen2.5:7b`, `mistral:7b` et `gemma2:9b` sur les métriques
de génération restent dans un intervalle de 0.05 à 0.10. **`mistral:7b` est
retenu pour la démo** car il offre le meilleur compromis
*answer relevancy / latence sur Together AI*.

## 6. Configuration recommandée

```
Embedding   : bge-m3
Retrieval   : pgvector (IVFFLAT, 10 probes), similarity_threshold = 0.5
HyDE        : ❌
Reranking   : ✅ (cross-encoder/ms-marco-MiniLM-L-6-v2)
Generation  : mistral:7b via Together AI
```

C'est la configuration utilisée par défaut dans la démo en ligne.

## 7. Limites connues de l'évaluation

- **Taille du dataset** (79 paires) : suffisante pour des tendances mais
  insuffisante pour des conclusions statistiquement significatives à fine
  granularité par sous-domaine.
- **Juge unique** : `gpt-4o-mini` peut introduire un biais systématique
  partagé par toutes les configurations.
- **Coût** : chaque run complet sur les 79 paires consomme entre 10k et 30k
  tokens du juge, ce qui limite le nombre de réplicats faisables.

## 8. Reproduire l'évaluation

```bash
# 1. Restaurer le corpus de démo
./scripts/dump_for_demo.sh

# 2. Lancer une configuration spécifique
python -m legal_ai.evaluation --hyde --rerank --generation-model mistral:7b

# 3. Les scores sont ajoutés à evals/summary.json
#    Le détail par question est dans evals/results/
```
