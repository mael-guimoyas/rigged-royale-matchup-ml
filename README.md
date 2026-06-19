# Rigged Royale Matchup ML

Pipeline Python pour estimer la **qualité intrinsèque d'un matchup Clash Royale** sans
modéliser le skill individuel des joueurs.

Le modèle apprend directement sur les vrais résultats `win/loss`. La matrice empirique est
utilisable comme prior, jamais comme cible principale. Le modèle prend en compte :

- les 16 cartes du matchup ;
- les héros et champions via l'identité de la carte et son rôle ;
- les évolutions de chaque carte ;
- les interactions apprises entre les 64 paires carte contre carte du matchup ;
- les synergies apprises entre les 28 paires de cartes de chaque deck ;
- un mini-Transformer optionnel sur les 16 cartes orientées `team/opponent` ;
- des adapters par segment qui modulent les interactions selon le contexte méta ;
- les tower troops ;
- le mode, le patch et le segment d'arène/ladder/ranked league ;
- facultativement, la probabilité fournie par la matrice existante.

Il n'utilise ni identité, ni historique, ni winrate personnel du joueur.
Les interactions et synergies ne sont pas hardcodées : elles sont apprises uniquement depuis
les résultats `win/loss`.

## Garanties importantes

- Déduplication des mêmes parties récupérées depuis plusieurs joueurs.
- Extraction Supabase en lecture seule et pagination par `(inserted_at, fingerprint)`, sans `OFFSET`.
- Fichiers Parquet compressés adaptés à plusieurs millions de combats.
- Découpage chronologique 70/15/15, jamais aléatoire.
- Augmentation par inversion des camps.
- Architecture antisymétrique : `P(A > B) = 1 - P(B > A)`.
- Calibration sur la validation et test futur totalement séparé.
- Rapport méta avec précision `bad/neutral/good`, couverture et taux correct/incorrect.

## Installation

Windows PowerShell :

```powershell
cd rigged-royale-matchup-ml
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
Copy-Item .env.example .env
```

Place dans `.env` l'URL Postgres affichée par **Supabase > Connect** :

```dotenv
SUPABASE_DB_URL=postgresql://USER:PASSWORD@HOST:5432/postgres?sslmode=require
```

Ce projet n'a pas besoin de clé `service_role`. Pour une extraction longue, utilise de
préférence l'URL **Session Pooler** et un utilisateur Postgres en lecture seule.

## Pipeline
### 0. Index d'extraction recommandé

Exécute une fois `sql/001_ml_extraction_index.sql` dans le SQL Editor Supabase. L'index est
créé avec `CONCURRENTLY` pour ne pas bloquer le collecteur. Il permet une pagination fiable
sur `(inserted_at, fingerprint)` : contrairement au hash `fingerprint`, `inserted_at` permet
de récupérer également tous les futurs combats.
Le script ajoute aussi un index partiel `ranked`/`ladder`, utilisé par l'extraction pour
éviter de télécharger des modes qui seront filtrés ensuite.

### 1. Test d'extraction

```powershell
rigged-matchup extract --max-rows 50000
```

Puis extraction complète :

```powershell
rigged-matchup extract
```

L'extraction utilise maintenant des batchs Postgres de `25000` lignes par défaut, une
déduplication SQLite en bulk, et récupère `leagueNumber` directement pendant la requête
quand il est présent. Pour tester un batch plus agressif :

```powershell
rigged-matchup extract --batch-size 50000
```

Si la mémoire monte trop ou si Supabase coupe la requête, reviens à `25000` ou `10000`.
Pour une extraction longue, préfère une connexion Postgres directe ou le Session Pooler
plutôt que la Data API REST.

L'extraction reprend après une interruption grâce à `data/extract-state.json`. Les doublons
sont suivis dans `data/dedup.sqlite3`.

Si des fichiers `data/raw/*.parquet` ont été extraits avant la séparation ranked par league,
mets-les à jour une fois avant de reconstruire les splits :

```powershell
rigged-matchup backfill-ranked-segments
```

Cette étape lit `leagueNumber` depuis Supabase via `source_fingerprint`. Les trophées ranked
ne sont jamais utilisés comme fallback : sans `leagueNumber`, le segment reste
`ranked:unknown`.

### 1b. Collecte directe depuis l'API Clash Royale (sans Postgres)

Pour alimenter l'entraînement sans stocker les combats bruts dans Postgres, `collect-api`
récupère les battlelogs directement depuis l'API CR et écrit des shards Parquet
prêts pour l'entraînement (mêmes colonnes que `extract`), localement et/ou dans
Supabase Storage.

Variables `.env` requises : `CR_API_TOKEN` (+ `CR_API_BASE_URL` = proxy RoyaleAPI). Pour
l'upload Storage : `SUPABASE_URL` et `SUPABASE_SECRET_KEY`. Copie-les depuis le repo app.

```powershell
# Test court : amorce 20 joueurs déjà connus dans Supabase, local seulement.
rigged-matchup collect-api --from-supabase 20 --max-players 50

# Prod : amorce 500 joueurs, étend via les adversaires, upload dans Storage.
rigged-matchup collect-api --from-supabase 500 --upload --max-battles 200000

# Ou amorce depuis un fichier de tags (un tag par ligne ; le fichier doit exister).
rigged-matchup collect-api --tags-file tags.txt --max-players 50
```

Il faut une amorce : `--from-supabase N` (tags depuis `public.players`) ou `--tags-file`.
Chaque battlelog fournit l'adversaire ; `--snowball` (par défaut) met ces tags en file
pour élargir la couverture. La `leagueNumber` vient du profil du joueur (`/players/{tag}`),
donc les combats ranked sont segmentés `ranked:league-N` dès la collecte, sans backfill.
La déduplication réutilise `data/dedup.sqlite3` (clé `game_id` canonique : un même combat
vu chez deux joueurs n'est compté qu'une fois).

Storage est du stockage objet : ces shards sont lus en masse par `prepare`/l'entraînement,
jamais ligne par ligne. C'est le bon support pour un corpus d'entraînement. Remplis Storage
d'abord, vérifie l'entraînement, puis seulement supprime `raw` de Postgres.

Sur une autre machine (ou après nettoyage local), récupère le corpus depuis Storage au lieu
de la base — c'est le pendant de `extract`, mais la source est Storage :

```powershell
rigged-matchup pull-storage --bucket training-battles --prefix battles
rigged-matchup prepare
```

### 2. Préparation chronologique

```powershell
rigged-matchup prepare
```

Pour reconstruire les splits :

```powershell
rigged-matchup prepare --overwrite
```

### 3. Entraînement

Option recommandée avant l'entraînement : attacher un prior empirique reconstruit
uniquement sur la période `train`. Cette étape utilise les règles de matrice du repo
app `../riggedroyale`, reconstruit la matrice depuis `data/prepared/train`, puis remplace
`matrix_prior` dans les splits préparés.

```powershell
rigged-matchup attach-prior
rigged-matchup train

```

Le rapport `artifacts/prior-metrics.json` contient les métriques du prior seul sur
validation/test, la couverture par niveau (`archGlobal`, `planBucket`, etc.) et les
métriques par segment. N'utilise pas `empirical_matchup_snapshots.current` pour un
benchmark passé si sa date de construction est postérieure au début de la validation
ou du test.

Les combats ranked sont segmentés par `leagueNumber` (`ranked:league-7`, etc.).
Les trophées ranked ne sont pas utilisés comme fallback de league ; sans `leagueNumber`,
le segment reste `ranked:unknown`.

```powershell
rigged-matchup train
```

Le checkpoint calibré est écrit dans `artifacts/matchup-model.pt`. Une carte graphique CUDA
est utilisée automatiquement si elle est disponible.

La matrice empirique est désactivée par défaut pendant cette phase
(`matrix_prior_strength: 0.0`). Même si une colonne `matrix_prior` existe, le modèle entraîné
ignore son logit tant que cette valeur reste à `0.0`.

Après sélection du meilleur checkpoint, une calibration affine est apprise sur la validation :
`logit / temperature + bias`, globalement puis par segment. Les segments avec moins de
`segment_calibration_min_rows` exemples utilisent la calibration globale. Le logit brut du
modèle reste antisymétrique ; le biais de calibration sert à corriger les probabilités
observées par segment.

`card_dropout` masque parfois une carte pendant l'entraînement uniquement. Cela force le
modèle à apprendre des structures de deck plus robustes au lieu de dépendre trop fortement
d'une composition exacte.

Le taux d'apprentissage suit un warmup linéaire (`warmup_fraction`) puis une décroissance
cosinus. `label_smoothing` adoucit légèrement les cibles vers `0.5` : comme un seul `win/loss`
est très bruité, cela réduit la surconfiance et améliore le Brier et la calibration.

L'interaction carte contre carte (`use_bilinear_cross`) n'est plus un simple produit terme à
terme symétrique : une matrice bilinéaire apprise `(A·W) ⊙ B` encode une relation orientée
« la carte A counter la carte B ». Elle est initialisée à l'identité et apprend la structure de
counter uniquement depuis les résultats.

Par défaut, l'entraînement utilise des micro-batches de `256` lignes avec
`gradient_accumulation_steps: 8`, soit un batch effectif de `2048`. Cela garde un débit
d'itérations élevé sans changer l'échelle d'optimisation. L'évaluation utilise un batch plus
large (`evaluation_batch_size: 4096`) car elle ne fait pas de rétropropagation.

### 4. Évaluation générale

```powershell
rigged-matchup evaluate artifacts/matchup-model.pt
```

Métriques : AUC, log-loss, Brier score, accuracy et erreur de calibration. Le rapport
contient désormais aussi :

- `expected_calibration_error_quantile` : ECE sur bins équi-effectifs, plus fiable que les
  bins largeur-égale quand les probabilités se tassent autour de `0.5` ;
- `calibration_slope` / `calibration_intercept` : recalibration logistique `cible ~ logit(p)`.
  Pente `1` = calibré ; pente `< 1` = surconfiant, `> 1` = sousconfiant ;
- `confidence_intervals` : intervalles bootstrap (95 %) sur AUC/Brier/log-loss/accuracy quand
  `evaluation.bootstrap_samples > 0`. Indispensable sur les petits splits.

### 4d. Benchmark et plancher de bruit

Les chiffres bruts (AUC ~0,57) n'ont de sens que comparés à des baselines. Cette commande écrit
un rapport unique qui place le modèle face à des références sur le même split :

```powershell
rigged-matchup benchmark artifacts/matchup-model.pt --split test --min-support 100
```

`artifacts/benchmark-test-report.json` compare :

- `constant_0.5` : un modèle qui ne sait rien ;
- `matrix_prior` : la matrice empirique seule ;
- `model` : le modèle calibré ;
- `noise_floor` : le **plancher de bruit irréductible**. Avec des labels `win/loss` binaires et
  un vrai taux de matchup `p`, le meilleur Brier possible par partie est `p*(1-p)`. Le taux `p`
  est estimé par le taux observé de chaque matchup ayant au moins `min-support` parties.

Le champ clé est `model_brier_gap_to_floor` : si le Brier du modèle sur les matchups supportés
est déjà proche de `irreducible_brier`, il reste peu de discrimination à extraire et l'effort
doit aller vers la couverture ou les données plutôt que l'architecture.
`brier_headroom_captured_vs_prior` indique la part de la marge prior→plancher que le modèle
capture.

### 4b. Évaluation stricte des matchups jamais vus

Ce test garde uniquement les combats du split choisi dont le matchup exact est absent de
`train`. La clé est canonique : `A vs B` et `B vs A` produisent la même `matchup_key`.

```powershell
rigged-matchup evaluate-unseen artifacts/matchup-model.pt --split test
```

Le rapport est écrit dans `artifacts/unseen-test-matchup-metrics.json`. Il contient plusieurs
niveaux de difficulté :

- `all_unseen_matchups` : tous les matchups exacts absents de `train` ;
- `known_decks_new_matchup` : les deux decks existent déjà séparément dans `train`, mais pas
  leur matchup ;
- `one_new_deck` : un seul des deux decks complets est absent de `train` ;
- `two_new_decks` : les deux decks complets sont absents de `train`.

Chaque sous-split est conservé dans `artifacts/unseen-test-matchups/<niveau>/data.parquet`
pour audit. Sur `all_unseen_matchups`, viser environ `auc > 0.55` et `brier_score < 0.25`
indique que le modèle généralise au-delà des matchups exacts déjà vus. Les niveaux
`one_new_deck` et `two_new_decks` sont volontairement plus difficiles.

### 4c. Diagnostic des segments ranked

Si trop de combats ranked restent dans `ranked:unknown`, vérifie la couverture locale et si
les fingerprints peuvent encore être retrouvés dans la base configurée :

```powershell
rigged-matchup diagnose-ranked-segments
```

Si `database_match_count` vaut `0` sur l'échantillon, les Parquet locaux ne correspondent pas
à la base Supabase pointée par `.env` ; `backfill-ranked-segments` ne pourra alors pas
réparer ces anciennes lignes.

### 5. Taux correct/incorrect dans la méta

```powershell
rigged-matchup evaluate-meta artifacts/matchup-model.pt --min-support 100
```

Le fichier `artifacts/meta-report.json` contient notamment :

- `meta_weighted_class_accuracy` : taux correct pondéré par la fréquence réelle ;
- `coverage` : fraction de la méta ayant au moins le support demandé ;
- `matchup_mae` : erreur moyenne entre probabilité annoncée et taux observé ;
- distribution `bad/neutral/good` dans la méta.

Le rapport exact-deck peut avoir une faible couverture. C'est normal : baisse
`--min-support` avec prudence ou ajoute ensuite un rapport par plan/archétype.

### 6. Prédiction

```powershell
rigged-matchup predict artifacts/matchup-model.pt example-matchup.json
```

Les rôles utilisent les codes suivants : `1=normal`, `2=champion`, `3=hero`. L'identité de
la carte reste toujours la variable principale ; un nouveau héros est donc supporté dès que
son ID apparaît dans les données d'entraînement.

## Brancher la matrice existante

Sans branchement, `matrix_prior=0.5` et le réseau apprend uniquement sur les cartes et les
résultats réels. C'est déjà valide.

Pour ajouter la matrice, crée une fonction Python :

```python
def predict(record: dict) -> float:
    # Classifier les decks avec les mêmes règles que l'application,
    # puis lire archBucket/planBucket/... dans le snapshot.
    return 0.63
```

Puis configure :

```yaml
data:
  matrix_prior_provider: "mon_module:predict"
```

Le réseau apprend alors un **résidu autour de la matrice**. Il peut la dépasser, car sa cible
reste le résultat réel et qu'il voit les cartes, héros et évolutions précis.

Important : ne branche pas un prior construit avec les données de validation/test. Pour une
évaluation incontestable, reconstruis la matrice avec la période d'entraînement uniquement.

## Niveau des cartes

Les niveaux API bruts ne sont pas directement comparables entre raretés. Ils ne sont donc
pas fournis au modèle. Si l'écart de niveau doit être neutralisé, configure
`max_raw_average_level_difference` après avoir ajouté une normalisation fiable par rareté,
ou entraîne prioritairement sur les modes aux niveaux standardisés.

## Structure

```text
src/rigged_matchup_ml/
  extraction.py       extraction Supabase et déduplication
  api_collect.py      collecte directe API CR vers Parquet local + Supabase Storage
  domain.py           parsing cartes, héros, évolutions et segments
  prepare.py          Parquet, splits temporels et vocabulaires
  empirical_prior.py  prior empirique train-only et diagnostics par segment
  model.py            encodeur de decks et réseau antisymétrique
  trainer.py          entraînement, warmup+cosine, label smoothing et calibration
  metrics.py          AUC, log-loss, Brier, calibration (slope, ECE quantile) et bootstrap
  benchmark.py        baselines, plancher de bruit irréductible et rapport comparatif
  meta_evaluation.py  taux correct/incorrect dans la méta
  predictor.py        inférence d'un matchup
```

## Validation attendue avec 3–4 millions de combats

Ne fixe pas un objectif uniquement sur l'accuracy d'une partie individuelle. Le critère
principal est la calibration sur une période future. Cibles raisonnables :

- AUC supérieure à la baseline empirique actuelle (~0,568) ;
- Brier et log-loss inférieurs à la matrice seule ;
- erreur de 2–4 points sur les matchups fréquents ;
- rapporter simultanément la précision et la couverture.
