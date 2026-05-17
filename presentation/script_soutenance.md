# Guide de maîtrise — Soutenance de Thèse Professionnelle

> Architecture RAG souveraine pour la capitalisation des connaissances — Cas du FART
> Format : **20 min de présentation** + 10-20 min de questions. Objectif : ~50 s par slide.

---

## 1. Répartition du temps (cible 20 min)

| Bloc | Slides | Durée | Cumul |
|------|--------|-------|-------|
| Introduction + problématique | 1-6 | 4 min | 4:00 |
| Ch.1 — État de l'art (à survoler) | 7-8 | 1 min 30 | 5:30 |
| Ch.2 — Concepts, méthodes, outils | 9-15 | 6 min | 11:30 |
| Ch.3 — Résultats | 16-21 | 6 min | 17:30 |
| Discussion + conclusion | 22-24 | 2 min 30 | 20:00 |

**Règle de survie** : si tu débordes, sacrifie les slides 7 (état de l'art) et 15 (conversation).
Ne sacrifie jamais 5 (problématique), 11 (splitter), 19 (synthèse ablation), 22 (réponse).

---

## 2. Chiffres à connaître par cœur

- **Jeu de test** : 79 paires Q/R · 3 domaines (social, fiscal, sociétés) × 3 profils (factuel,
  synthèse multi-articles, hors-corpus).
- **Plan d'expérience** : 12 configurations = 3 modèles × HyDE (on/off) × reranking (on/off).
- **Paramètres fixes** : embedding `bge-m3` · reranker `ms-marco-MiniLM-L-6-v2` · top_k 6 ·
  seuil cosinus 0,5 · pgvector IVFFLAT (100 listes, 10 probes). Juge RAGAS : `gpt-4o-mini`.
- **Reranking** : +5 à 8 pts de **précision du contexte** (systématique).
- **HyDE seul** : **dégrade** la précision (~−8 pts).
- **Config retenue** : sans HyDE + reranking → mistral **0,790**, qwen 0,782, gemma 0,757.
- **Profils des modèles** : qwen2.5:7b = fidélité **0,709** (max) ; mistral:7b = pertinence
  **0,835** (max).
- **5 articles** : Lewis 2020 (RAG) · Reimers & Gurevych 2019 (SBERT) · Gao 2022 (HyDE) ·
  Nogueira & Cho 2019 (reranking) · Es 2023 (RAGAS).

---

## 3. Script slide par slide

> Le texte en « … » est à dire à l'oral (reformule avec tes mots). **T** = transition.

### Slide 1 — Titre (20 s)
« Madame, Monsieur les membres du jury, je vous présente ma thèse professionnelle :
la conception et l'évaluation d'une architecture RAG souveraine pour la capitalisation des
connaissances, appliquée au cas du FART. »
**T** : « Je vais d'abord poser le contexte et la problématique, puis dérouler les trois chapitres. »

### Slide 2 — Plan (20 s)
Annonce les 6 temps. Ne lis pas tout : « Introduction et problématique, puis revue de littérature,
concepts et méthodes, résultats, et enfin discussion et conclusion. »

### Slide 3 — Contexte (45 s)
« La transformation numérique a produit une surcharge informationnelle. Le FART détient un
patrimoine documentaire critique — sa mémoire institutionnelle — composé de normes juridiques, de
documents procéduraux et décisionnels. Mais ce patrimoine est fragmenté, souvent en PDF non
structurés ou scannés. »

### Slide 4 — Problème central + contraintes (1 min)
« Le diagnostic : pas d'indexation centralisée, des silos individuels, une recherche sémantique
impossible, et une dépendance au savoir tacite qui s'érode avec la mobilité du personnel.
Trois contraintes encadrent toute solution : la **souveraineté** — tout doit rester local ;
le **risque hallucinatoire** — en droit, le système doit être fidèle et citer ses sources ;
et l'**hétérogénéité** — beaucoup de scans, donc OCR et segmentation. »

### Slide 5 — Problématique (40 s) ⭐
Lis la question **lentement**, mot à mot. Puis : « Il ne s'agit pas de déployer un outil, mais de
concilier innovation cognitive et gouvernance des données. »

### Slide 6 — Axes & hypothèses (45 s)
« Deux axes. Le premier, technique : quelle configuration de modèles open-source en local — d'où
l'hypothèse H1 sur l'apport de HyDE et du reranking. Le second, méthodologique : comment évaluer
objectivement via RAGAS — d'où H2 sur la suffisance des modèles locaux. »
**T** : « Ces choix s'appuient sur cinq travaux. »

### Slide 7 — État de l'art (50 s) — SURVOLER
« Je passe vite, le jury a lu le chapitre. Cinq travaux en fil conducteur : Lewis pose le RAG ;
Reimers et Gurevych, les embeddings ; Gao, HyDE ; Nogueira et Cho, le reranking ; Es, l'évaluation
RAGAS. Chaque brique de mon système découle de l'un d'eux. »

### Slide 8 — Positionnement (40 s)
« Mon originalité : la souveraineté totale en local, l'intégration combinée HyDE + reranking
évaluée empiriquement, et la spécialisation juridique marocaine. »
**T** : « Voyons maintenant l'architecture concrète. »

### Slide 9 — Architecture (50 s)
« Cinq phases séquentielles, découplées par une base de données : ingestion, extraction par
Docling, indexation, recherche, conversation. Chaque composant suit une interface abstraite, donc
les modèles sont interchangeables. Les phases 3 et 4 portent mes contributions. »

### Slide 10 — Segmentation (50 s)
« Le Bulletin Officiel a une hiérarchie — Dahir, Loi, Titre, Chapitre, Article — portée par la
typographie, non balisée. Mon splitter la normalise avant de découper en chunks de 1500 caractères,
et ajoute à chaque chunk un fil d'Ariane pour la traçabilité. »
**T** : « Concrètement, voici l'algorithme. »

### Slide 11 — Algorithme du splitter (1 min) ⭐ PRATIQUE
« Voici le cœur, la fonction `_classify` que j'ai écrite. Pour chaque titre : un article devient du
gras ; un mot-clé légal reçoit un niveau fixe via la table de droite ; un titre libre est inféré du
contexte. En bas, l'exemple : Docling sort Loi en H1 et Article en titre — incohérent ; mon
algorithme réécrit selon la hiérarchie juridique. C'est ce qui rend les breadcrumbs fiables. »

### Slide 12 — Recherche HyDE + reranking (50 s)
« Pour combler le fossé lexical entre la question courante et le corpus normatif, HyDE génère une
réponse hypothétique et lance deux récupérations fusionnées. Puis le cross-encoder ms-marco
réordonne finement les candidats : c'est le retrieve-then-rerank. Ces deux leviers sont les
variables de mon ablation. »

### Slide 13 — Modèles candidats (45 s)
« La souveraineté impose des modèles open-source locaux. Embedding de référence : bge-m3,
multilingue. Génération : qwen2.5:7b, comparé à mistral et gemma. Reranker : ms-marco, assez léger
pour le CPU. Stockage : pgvector. **Important : embedding et reranker sont fixés ; seule la
génération est comparée.** »

### Slide 14 — Évaluation & jeu de test (1 min) ⭐
« RAGAS mesure séparément le retrieval et la génération via quatre métriques. Le jeu de test est
construit en trois étapes : extraction de chunks seed, génération synthétique par `gpt-4o-mini` —
volontairement distinct des modèles évalués pour éviter le biais d'auto-évaluation — puis
validation humaine de chaque paire. »

### Slide 15 — Conversation (35 s) — peut être survolée
« Pour les questions de suivi, je maintiens un historique ; au-delà de 2000 tokens, le LLM le
résume et garde les 4 derniers messages. Simple, sans dépendance externe. »
**T** : « Passons aux résultats. »

### Slide 16 — Protocole (50 s)
« 79 paires stratifiées sur 3 domaines et 3 profils, dont un profil hors-corpus qui teste le refus.
12 configurations. Paramètres fixes — bge-m3, ms-marco, top_k 6, seuil 0,5. Juge gpt-4o-mini, hors
production. »

### Slide 17 — Exemples du jeu de test (1 min) ⭐ PRATIQUE
« Trois vraies questions, une par profil. Une factuelle — un montant budgétaire, article 29. Une
de synthèse — la fusion d'OPCVM, article 78. Et surtout une hors-corpus : le système répond
correctement qu'il n'a pas l'information, au lieu d'inventer. C'est le test d'anti-hallucination,
central en droit. »

### Slide 18 — Ablation : reranking + HyDE (1 min)
« Le reranking seul gagne 5 à 8 points de précision, systématiquement. HyDE seul, à l'inverse,
**dégrade** la précision d'environ 8 points : le modèle généraliste paraphrase en français courant
au lieu du registre normatif, ce qui éloigne le vecteur du corpus. C'est l'effet contre-intuitif
central de mes résultats. »

### Slide 19 — Synthèse ablation (1 min 15) ⭐ POINT CHAUD
« Sur les 12 configurations : en score brut, HyDE+reranking est marginalement le plus haut pour
qwen, à 0,794. Mais HyDE est instable et coûte une seconde passe d'embedding — 30 à 50 % de latence
en plus. Je retiens donc **sans HyDE + reranking** : gain systématique, coût maîtrisé, meilleure
précision. **H1 est donc validée partiellement** : le reranking confirme, HyDE infirme. »

### Slide 20 — Comparaison des modèles (50 s)
« À configuration fixée, mistral a le meilleur score moyen et la meilleure pertinence ; qwen la
meilleure fidélité ; gemma est en retrait. La précision du contexte est stable, car le retrieval
est commun. »

### Slide 21 — Interprétation (50 s)
« Le point clé : une tension structurelle fidélité / pertinence qu'aucun modèle ne résout. D'où un
choix par profil de risque : qwen pour les usages à effet juridique direct, mistral pour
l'exploration. Des modèles locaux 7-9B atteignent 0,75 à 0,79 : **H2 est validée, sous arbitrage.** »

### Slide 22 — Réponse à la problématique (50 s) ⭐
« Je réponds donc à la problématique : oui, une architecture RAG souveraine permet une
capitalisation fiable et une restitution sourcée de la mémoire institutionnelle. La fidélité est
maîtrisée par le choix de modèle, et l'exécution est 100 % locale. »

### Slide 23 — Déploiement FART (45 s)
« Côté déploiement : aucun appel cloud en production ; seule l'évaluation utilise gpt-4o-mini.
Le Bulletin Officiel étant un corpus de substitution équivalent, les composants sont transposables ;
deux écarts restent à combler — adapter le splitter et reconstruire le jeu de test — soit une
mission d'environ 4 à 6 mois. »

### Slide 24 — Conclusion (40 s)
« En résumé : un pipeline souverain complet, un splitter juridique, une étude d'ablation et un
protocole d'évaluation. Les limites sont assumées — monolingue, jeu de test borné. Les
perspectives : fine-tuning juridique marocain, extension bilingue français-arabe, et déploiement
effectif au FART. Je vous remercie et reste à votre disposition. »

---

## 4. Questions du jury — réponses préparées

**« Pourquoi ne pas retenir HyDE + reranking, qui a le meilleur score brut (0,794) ? »**
→ L'écart est marginal (0,794 vs 0,782) et porté par un seul modèle. HyDE **dégrade la précision**
sur toutes les configs (il est instable) et impose une **seconde passe d'embedding** (+30-50 % de
latence). Le reranking seul donne un gain **systématique** sur les 3 modèles à coût maîtrisé.
En contexte institutionnel local, la stabilité et le coût priment sur 1 point de score.

**« Pourquoi HyDE échoue-t-il alors que la littérature le valide ? »**
→ HyDE suppose une homogénéité de registre entre le document hypothétique et le corpus. Un modèle
généraliste 7B paraphrase en français courant, pas dans le registre normatif du Bulletin Officiel.
Le vecteur hypothétique s'éloigne donc des passages cibles. Le fine-tuning lèverait cette limite.

**« La fidélité plafonne à ~0,70 — n'est-ce pas insuffisant en droit ? »**
→ C'est la limite des modèles locaux 7-9B sans fine-tuning, un choix assumé pour la souveraineté.
D'où la recommandation : qwen (fidélité maximale) pour les usages à effet juridique direct, et un
système qui **cite ses sources** (breadcrumbs) pour que l'utilisateur vérifie.

**« Le juge gpt-4o-mini est aussi le générateur du jeu de test : n'est-ce pas circulaire ? »**
→ Le générateur produit un brouillon, mais **chaque paire est validée à la main** : le
ground-truth final est humain. Et le juge est **distinct des modèles évalués** (qwen, mistral,
gemma). La revue humaine lève la circularité. C'est un compromis méthodologique, hors production.

**« Pourquoi pas de comparaison des embeddings ou des rerankers ? »**
→ Choix de périmètre lié aux contraintes matérielles locales : un embedding multilingue stable
(bge-m3) et un reranker léger (ms-marco) ont été fixés comme références justifiées. La variable
expérimentale retenue est le modèle de génération, le plus déterminant pour la fidélité.

**« Comment passez-vous du Bulletin Officiel au vrai corpus du FART ? »**
→ Équivalence structurelle et linguistique. Composants transposables (embedding, reranker,
génération, RAGAS). Deux écarts : adapter le splitter à la terminologie FART, et reconstruire un
jeu de test interne avec le même protocole. Effort estimé : 4 à 6 mois.

---

## 5. Plan d'entraînement (3 passes)

1. **Passe 1 — contenu (lecture)** : relis ce script + les notes orateur du PPTX (volet
   Commentaires). Vérifie que tu comprends *chaque chiffre* de la section 2.
2. **Passe 2 — chrono** : récite à voix haute, minuteur en main. Note les slides où tu dépasses.
   Vise 20 min ± 1. Coupe le superflu sur 7 et 15.
3. **Passe 3 — questions** : fais-toi poser les 6 questions de la section 4 par quelqu'un (ou à
   voix haute). Réponds en < 1 min chacune, sans notes.

**Le jour J** : respire sur la slide 5 (problématique), ralentis sur 11, 19 et 22 (tes temps forts).
Si une question te déstabilise, reformule-la avant de répondre — ça te donne 3 secondes pour penser.

---

*Notes orateur détaillées : disponibles dans le volet « Commentaires » de chaque slide du PPTX.*
