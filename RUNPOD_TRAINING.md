# Réentraînement RunPod depuis zéro

Prérequis : toutes les modifications ont été commit et push sur `main`, et les
shards sont déjà présents dans Supabase Storage. Il n'y a aucune nouvelle collecte
Clash Royale à lancer.

## 1. Créer le Pod

Configuration conseillée :

- 1 × RTX 5090 32 Go ;
- template officiel RunPod PyTorch compatible RTX 5090, avec CUDA et Python 3.11+ ;
- au moins 8 vCPU et 32 Go de RAM ;
- container disk : 30-40 Go ;
- volume disk : 80 Go ;
- terminal web/JupyterLab activé.

Le trainer utilise un seul GPU.

Sur cette configuration, prévoir environ **45 minutes pour l'entraînement**. Le
téléchargement Supabase, `prepare`, Card2Vec, l'évaluation et le benchmark s'ajoutent
à cette durée et dépendent surtout du réseau, du CPU et du volume de shards.

## 2. Tout lancer depuis un seul terminal web

Ouvrir le terminal RunPod, puis exécuter les commandes suivantes dans ce même
terminal :

```bash
cd /workspace
git clone https://github.com/mael-guimoyas/rigged-royale-matchup-ml.git
cd rigged-royale-matchup-ml

git log -1 --oneline
python --version
nvidia-smi

# Ne pas enregistrer la clé Supabase dans l'historique Bash.
set +o history
export SUPABASE_URL='https://PROJECT_REF.supabase.co'
export SUPABASE_SECRET_KEY='COLLER_LA_CLE_SECRETE_ICI'
set -o history

test -n "$SUPABASE_URL" && test -n "$SUPABASE_SECRET_KEY"

export TRAINING_BUCKET=training-battles
export TRAINING_PREFIX=battles
export STORAGE_DOWNLOAD_WORKERS=16

export RUNPOD_BATCH_SIZE=8192
export RUNPOD_EVAL_BATCH_SIZE=16384
export RUNPOD_GRAD_ACCUM=1
export RUNPOD_NUM_WORKERS=8
export RUNPOD_EPOCHS=15
export RUN_ATTACH_PRIOR=0
export RUN_BENCHMARK=1

bash scripts/runpod_train.sh
```

Remplacer les deux valeurs d'exemple avant d'exécuter le bloc. Ces variables existent
uniquement dans le terminal courant et ne viennent jamais de Git. Sur le terminal
Linux RunPod, la bonne syntaxe est `export NOM=valeur` : `set NOM=valeur` est une
syntaxe Windows et ne transmettrait pas la variable au script.

`SUPABASE_SECRET_KEY` doit rester une variable serveur : ne pas la mettre dans le
frontend, un fichier commité ou une capture d'écran.

Le script fait directement toute la chaîne :

```text
Supabase Storage
  -> téléchargement des shards
  -> préparation chronologique train/validation/test
  -> préentraînement Card2Vec
  -> entraînement CUDA
  -> calibration et évaluation
  -> benchmark
  -> /workspace/artifacts/matchup-model.pt
```

Les anciens shards sont compatibles : les flags Hero/Evo sont corrigés pendant
leur lecture. Il n'y a ni recollecte ni migration à faire.

`RUN_ATTACH_PRIOR=0` est volontaire : le prior nécessite aussi le dépôt web
`riggedroyale`. Le modèle est actuellement configuré pour apprendre sans ce prior.

## 3. Vérifier le résultat

Lorsque le script affiche `Done. Artifacts are in /workspace/artifacts` :

```bash
source /workspace/rrm-venv/bin/activate

python - <<'PY'
import json
from pathlib import Path

m = json.loads(Path('/workspace/data/prepared/manifest.json').read_text())
print('raw_rows:', m['raw_rows'])
print('unique_rows:', m['unique_rows'])
print('splits:', m['counts'])
print('cards_absent_from_train:', m['cards_absent_from_train'])
PY

ls -lh /workspace/artifacts
test -s /workspace/artifacts/matchup-model.pt
sha256sum /workspace/artifacts/matchup-model.pt
```

Les résultats principaux sont :

- `/workspace/artifacts/matchup-model.pt` ;
- `/workspace/artifacts/train.log` ;
- `/workspace/artifacts/evaluate.log` ;
- `/workspace/artifacts/benchmark.log`.

## 4. En cas de CUDA OOM

Réduire les batches et relancer :

```bash
export RUNPOD_BATCH_SIZE=4096
export RUNPOD_EVAL_BATCH_SIZE=8192
bash scripts/runpod_train.sh
```

Les shards, les splits et Card2Vec déjà présents sont réutilisés.

## 5. Télécharger le modèle

Dans le même terminal :

```bash
tar -czf /workspace/runpod-result.tar.gz -C /workspace/artifacts .
ls -lh /workspace/runpod-result.tar.gz
```

Dans JupyterLab, ouvrir `/workspace`, faire un clic droit sur
`runpod-result.tar.gz`, puis **Download**.

Après avoir vérifié l'archive téléchargée, arrêter ou terminer le Pod pour couper
la facturation.
