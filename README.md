# Rigged Royale Matchup ML

Pipeline Python pour estimer la **qualité intrinsèque d'un matchup Clash Royale** sans
modéliser le skill individuel des joueurs.

Le modèle apprend directement sur les vrais résultats `win/loss`. La matrice empirique est
utilisable comme prior, jamais comme cible principale. Le modèle prend en compte :

- les 16 cartes du matchup ;
- les héros et champions via l'identité de la carte et son rôle ;
- les évolutions de chaque carte ;
- les tower troops ;
- le mode, le patch et le segment d'arène/ladder/ranked league ;
- facultativement, la probabilité fournie par la matrice existante.

Il n'utilise ni identité, ni historique, ni winrate personnel du joueur.

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

### 1. Test d'extraction

```powershell
rigged-matchup extract --max-rows 50000
```

Puis extraction complète :

```powershell
rigged-matchup extract
```

L'extraction reprend après une interruption grâce à `data/extract-state.json`. Les doublons
sont suivis dans `data/dedup.sqlite3`.

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

### 4. Évaluation générale

```powershell
rigged-matchup evaluate artifacts/matchup-model.pt
```

Métriques : AUC, log-loss, Brier score, accuracy et erreur de calibration.

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
  domain.py           parsing cartes, héros, évolutions et segments
  prepare.py          Parquet, splits temporels et vocabulaires
  empirical_prior.py  prior empirique train-only et diagnostics par segment
  model.py            encodeur de decks et réseau antisymétrique
  trainer.py          entraînement, early stopping et calibration
  metrics.py          AUC, log-loss, Brier et calibration
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
