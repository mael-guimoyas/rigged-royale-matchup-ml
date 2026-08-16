# Ensemble RunPod — entraîner plusieurs graines

Complément à `RUNPOD_TRAINING.md`, qui reste la référence pour un entraînement
unique. Ce document décrit uniquement ce qui change pour produire un ensemble.

## Pourquoi

L'optimiseur retient le meilleur score parmi des milliers de decks que personne
n'a jamais joués. Le maximum de N estimations bruitées tombe sur le deck que le
modèle surestime le plus : mesuré sur des decks réels vérifiables, passer de 1 à
100 candidats fait monter le score promis de 0,0752 alors que le taux réellement
observé ne monte que de 0,0123. Environ 16 % de ce que la recherche annonce est
réel.

Aucun détecteur au niveau du deck ne peut corriger ça — paires, triplets,
groupes de quatre cartes, règles de composition, planchers de support et
MC-dropout ont tous été mesurés et écartés — parce que les decks émis ne sont pas
anormaux. C'est le chiffre qui l'est.

Des modèles entraînés indépendamment divergent précisément là où les données ne
les contraignent pas. Ça donne deux choses :

- **une correction d'affichage exacte** : sélectionner avec le membre A et
  afficher le score du membre B. L'erreur de B est indépendante du choix de A,
  donc le nombre affiché est non biaisé par construction, à n'importe quel nombre
  de candidats et sans avoir à estimer quoi que ce soit. Deux membres suffisent ;
- **une incertitude par deck**, qui varie réellement d'un candidat généré à
  l'autre, contrairement à la table par support qui leur attribuait à tous la
  même valeur.

## Faut-il collecter plus de batailles ? Non

Un ensemble réentraîne **le même corpus** avec des graines différentes ; c'est la
variabilité d'initialisation qu'on cherche à capturer. Les coupures
chronologiques 70/15/15 doivent rester rigoureusement identiques d'un membre à
l'autre, sans quoi leurs scores ne sont plus comparables et la validation hors
échantillon ne veut plus rien dire. Aucune collecte, aucune migration.

Le script garantit ça en ne faisant le téléchargement Supabase, `prepare` et
`card2vec` qu'une seule fois, à l'intérieur du premier membre ; tous les suivants
réutilisent `/workspace/data`.

À noter : `card2vec` est un PPMI+SVD déterministe sur les co-occurrences, sans
aucun aléa. Il produira les mêmes vecteurs quelle que soit la graine, donc tous
les membres partagent le même point de départ pour les plongements de cartes. La
diversité vient de l'initialisation des couches supérieures, de l'ordre des lots
et des masques de dropout. Si le désaccord mesuré s'avère trop faible, le levier
suivant est de passer `card2vec_init: false` sur deux membres — ça ajoute de la
diversité réelle, à un coût de qualité individuelle qu'il faudra alors mesurer.

## 1. Le Pod

Identique à `RUNPOD_TRAINING.md` : 1 × RTX 5090 32 Go, template PyTorch CUDA
Python 3.11+, 8 vCPU, 32 Go de RAM, container disk 30-40 Go, volume disk 80 Go,
terminal web activé.

Budget temps, sur la base des 45 minutes par entraînement mesurées pour un
membre : compter environ **30 à 45 minutes une seule fois** pour le
téléchargement, `prepare` et `card2vec`, puis **45 minutes par graine**. Quatre
graines nouvelles font donc de l'ordre de 3 h 45 à 4 h de Pod.

## 2. Lancer

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

# Les quatre nouvelles graines. Le checkpoint actuel (graine 42) est le
# cinquième membre : il n'est pas réentraîné.
export SEEDS="101 202 303 404"
export RUN_BENCHMARK=0

bash scripts/runpod_ensemble.sh
```

`RUN_BENCHMARK=0` est volontaire : le benchmark complet n'apporte rien par
membre, et l'évaluation reste active, ce qui suffit à prouver que chaque membre
est individuellement sain. Si un membre affiche une AUC nettement en dessous des
autres, c'est un entraînement raté et il ne faut pas le mettre dans l'ensemble.

Le script est reprenable : un membre dont `matchup-model.pt` existe déjà est
sauté, donc un Pod interrompu se relance avec la même commande.

## 3. Vérifier avant de télécharger

```bash
ls -lh /workspace/ensemble/seed-*/matchup-model.pt
sha256sum /workspace/ensemble/seed-*/matchup-model.pt
grep -h "val_auc\|validation" /workspace/ensemble/seed-*/evaluate.log | tail -20
```

**Les quatre empreintes SHA-256 doivent être différentes.** Deux membres
identiques signifieraient que `RUNPOD_SEED` n'est pas arrivé jusqu'au trainer, et
tout le run serait sans valeur. Le script les affiche déjà à la fin.

## 4. Rapatrier

```bash
tar -czf /workspace/ensemble.tar.gz -C /workspace/ensemble .
ls -lh /workspace/ensemble.tar.gz
```

Puis clic droit → Download dans JupyterLab, et **arrêter le Pod** une fois
l'archive vérifiée.

En local, déposer les membres à côté du checkpoint actuel :

```
artifacts/ensemble/seed-42/matchup-model.pt   (copie de artifacts/matchup-model.pt)
artifacts/ensemble/seed-101/matchup-model.pt
artifacts/ensemble/seed-202/matchup-model.pt
artifacts/ensemble/seed-303/matchup-model.pt
artifacts/ensemble/seed-404/matchup-model.pt
```

## 5. La mesure qui décide de la suite

À faire dès que **deux** membres existent, sans attendre les autres :

```
PYTHONPATH=src %LOCALAPPDATA%\rigged-matchup-cuda-venv\Scripts\python.exe ^
  scripts/ensemble_disagreement.py ^
  --checkpoints artifacts/ensemble/seed-42/matchup-model.pt ^
              artifacts/ensemble/seed-101/matchup-model.pt
```

Le script répond à trois questions et écrit `artifacts/ensemble-disagreement.json`.

**Question 1 — l'écart entre membres.** Comparé sur les decks que l'optimiseur a
réellement émis et sur des decks réels. Si l'écart sur les decks émis est du même
ordre que σ = 0,127, l'ensemble a de quoi travailler. S'il est très inférieur,
les membres partagent leur angle mort et les graines supplémentaires
n'apporteront presque rien : arrêter là et se rabattre sur la correction en
ln(N).

**Question 2 — l'écart prédit-il l'erreur ?** Vérifié sur les decks réels, dont
le taux observé est connu. C'est le test exact sur lequel MC-dropout a échoué
(corrélation +0,065, plate par quintile). Le rapport contient la même table par
quintile : **une incertitude utilisable doit y être monotone**, pas seulement
corrélée. Si elle ne l'est pas, l'écart ne sert pas à pondérer la sélection — mais
la correction d'affichage du point 3 reste valable, elle ne dépend pas de ça.

**Question 3 — ce que la séparation change.** Le script rejoue la sélection avec
le membre A et rapporte le score du membre B pour le deck retenu. `gap_removed`
est le nombre de points d'inflation que la séparation retire. Contrôle intégré :
lancé avec deux fois le même checkpoint, il renvoie exactement `0,0000`, ce qui
vérifie que la mesure n'invente rien.

## 6. Ce qu'on en fait ensuite

Selon le résultat de la question 1, l'un ou l'autre :

- **écart large** : servir la moyenne d'ensemble pour la sélection (elle réduit
  le bruit que l'argmax récolte, donc elle améliore le deck choisi), et afficher
  le score d'un membre tenu à l'écart. La correction intérimaire en ln(N) devient
  inutile et doit être retirée.
- **écart étroit** : garder un seul modèle pour la sélection, mais utiliser
  quand même le second pour l'affichage, puisque la séparation ne coûte rien et
  supprime le biais exactement.

Dans les deux cas le service ML doit exposer les scores par membre, pas seulement
la moyenne : le site a besoin de A pour choisir et de B pour afficher.
