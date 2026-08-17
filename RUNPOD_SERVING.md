# Inférence GPU sur RunPod Serverless

Ce guide déploie le chemin d'inférence lourd (`/predict/batch`) sur un GPU
serverless RunPod, en gardant les prédictions unitaires sur le conteneur CPU
netcup existant derrière `model.riggedroyale.com`.

La configuration CPU est intacte : `Dockerfile` et `docker-compose.yml` n'ont pas
changé et restent ce que netcup exécute. Le GPU passe par `Dockerfile.gpu` et
`docker-compose.gpu.yml`, ajoutés à côté.

## L'architecture visée : deux déploiements, pas un

Les deux formes de requêtes ont des profils de coût opposés.

- `/predict` porte **deux lignes**. Le temps est dominé par l'aller-retour
  réseau, pas par le calcul. Un GPU distant la rendrait plus lente, et un
  démarrage à froid la rendrait beaucoup plus lente.
- `/predict/batch` porte **des centaines à des milliers de lignes**. C'est du
  calcul réel, amorti sur un seul aller-retour. C'est là que le GPU paie.

Le site sait maintenant router les deux séparément : `ML_INFERENCE_URL` reste
netcup et sert les prédictions unitaires et `/health`, tandis que
`ML_BATCH_INFERENCE_URL` envoie les balayages vectorisés vers RunPod. Sans cette
seconde variable, tout continue d'aller sur netcup et rien ne change.

Conséquence directe sur le démarrage à froid : **aucune fiche joueur ne peut être
bloquée par un worker GPU endormi**, puisque le chemin unitaire ne le touche
jamais. Un démarrage à froid ne retarde qu'un balayage de panel, lequel appelle
déjà `warmUpAllModels()` avant l'éventail de requêtes.

## Ce que le GPU va apporter, mesuré

Avant d'ajouter du matériel, l'encodeur de lignes a été vectorisé — il coûtait
plus cher que le modèle lui-même. Sur un lot de 512 lignes, il est passé de
144,5 ms à 6,6 ms (**22×**), et sur 2048 lignes de 1091,7 ms à 34,5 ms
(**31,6×**). Le chemin batch décode désormais les deux sens en une seule passe
et fait tourner le modèle sur un lot empilé de 2N lignes au lieu de deux lots de
N.

Répartition après ce travail, sur 512 lignes (CPU 6 threads) :

| Étape | Avant | Après |
| --- | --- | --- |
| Encodage (deux sens) | 144,5 ms | 6,6 ms |
| Modèle + sortie | ~210 ms | ~210 ms |
| Part du modèle dans le total | 59 % | **97 %** |

C'est le point important : le goulot n'est plus l'encodeur Python mais le
modèle, et le modèle est ce que le GPU accélère. Avant vectorisation, le gain
maximal atteignable par un GPU était plafonné à environ 3× par la loi d'Amdahl ;
il ne l'est plus.

### Débit mesuré, GPU contre CPU

Mesuré sur une **RTX 4050 Laptop** — un GPU d'entrée de gamme, volontairement,
puisque c'est la classe de matériel qu'il faut louer pour ce modèle. CPU de
référence : la même machine, 6 threads (donc déjà plus favorable que le
conteneur netcup, qui tourne avec `OMP_NUM_THREADS=1`).

| Lignes par lot | CPU 6 threads | RTX 4050 | Gain |
| --- | --- | --- | --- |
| 512 | 216 ms · 2 365 l/s | 27 ms · 18 928 l/s | 8,0× |
| 1024 | — | 48 ms · 21 154 l/s | — |
| 2048 | 838 ms · 2 444 l/s | 94 ms · 21 754 l/s | 8,9× |
| 4096 | — | 191 ms · 21 425 l/s | — |

**Balayage de 100 000 matchups**, en tranches de 2048 : **4,81 s** d'inférence
(20 769 lignes/s), pic mémoire GPU **552 Mio**. Le même balayage coûte environ
42 s sur le CPU à 6 threads, et davantage sur netcup.

Le débit plafonne dès 1024 lignes et **ne monte plus** : à 8192 lignes par
tranche il redescend à 19 109 lignes/s pour 2 173 Mio de VRAM. La bonne taille de
tranche est donc **2048**, pas la plus grande possible — c'est ce qui fixe le
défaut de `MAX_BATCH_REQUESTS`.

Rejouer ces mesures, sur n'importe quel device :

```bash
python scripts/serving_benchmark.py --device cpu --sizes 512,2048 --compare-legacy
python scripts/serving_benchmark.py --device cuda --sweep 100000 --chunk 2048
python scripts/startup_profile.py --device cuda
```

## Optimiser le démarrage à froid

FlashBoot supprime une partie de la latence autour du conteneur (planification,
pull d'image déjà vue), mais il ne supprime pas ce qui se passe **dans** le
conteneur à chaque démarrage. Ce qui suit est déjà en place dans
`Dockerfile.gpu` :

- **Image minimale.** Les dépendances déclarées du paquet incluent duckdb,
  pyarrow, psycopg et scikit-learn, qui n'existent que pour l'entraînement et la
  préparation des shards. L'image GPU les saute (`pip install --no-deps`) et
  n'installe que torch, numpy, FastAPI et uvicorn. Pour que ce soit possible,
  `dataset.py` importe pyarrow paresseusement, à l'intérieur des deux fonctions
  d'entraînement qui lisent du Parquet — le chemin d'import du service ne le
  touche donc jamais (~100 ms de moins, et une dépendance en moins dans l'image).
- **Pas d'image de base CUDA.** La roue torch CUDA embarque ses propres
  bibliothèques CUDA ; seul le pilote hôte est nécessaire, et il est injecté par
  le runtime de conteneurs. `python:3.11-slim` suffit, et l'image est d'autant
  plus petite.
- **Bytecode précompilé.** Sans `compileall` au build, le premier import compile
  les milliers de fichiers `.py` de torch à chaque démarrage à froid, dans une
  couche jetée à l'arrêt du worker.
- **`CUDA_MODULE_LOADING=LAZY`.** Les noyaux sont chargés à l'usage plutôt qu'à
  la création du contexte.
- **Chauffe au démarrage** (`MODEL_WARMUP=1`). Un lot à vide paye la création du
  contexte CUDA, le chargement des noyaux et l'allocation des espaces de travail
  cuBLAS avant que le répartiteur n'envoie du vrai trafic. Sur un worker qui
  descend à zéro, c'est un coût récurrent, pas ponctuel.

Ce que RunPod contrôle et qu'il faut régler dans la console :

- **Activer FlashBoot** sur l'endpoint.
- **Idle timeout** assez long pour couvrir un balayage entier, sinon le worker
  s'éteint entre deux lots et chaque reprise repaye un démarrage.
- **Un worker actif** (`min workers = 1`) supprime complètement le problème, au
  prix d'une facturation continue. C'est le seul réglage qui garantit un
  démarrage à froid nul ; FlashBoot le réduit, il ne l'annule pas.

### Décomposition mesurée du démarrage

`scripts/startup_profile.py` sur la RTX 4050, après les optimisations ci-dessus :

| Étape | Durée | Part |
| --- | --- | --- |
| `import torch` | 1 754 ms | 68 % |
| `import rigged_matchup_ml` | 279 ms | 11 % |
| Création du contexte CUDA | 117 ms | 5 % |
| Chargement du checkpoint | 29 ms | 1 % |
| Chauffe | 280 ms | 11 % |
| Premier lot de 2048 | 106 ms | 4 % |
| **Total jusqu'à la première prédiction** | **2 566 ms** | |

**2,6 secondes** côté conteneur, très en dessous de la cible de 30 s. Le reste
d'un démarrage à froid RunPod est de la planification et du pull d'image, ce que
FlashBoot est précisément là pour absorber.

`import torch` domine à lui seul et n'est pas compressible : c'est du code
tiers. C'est aussi pourquoi la précompilation du bytecode compte — sans elle,
cet import recompile des milliers de fichiers à chaque démarrage à froid.

À relancer une fois sur le worker RunPod : les chiffres ci-dessus viennent d'une
machine Windows au cache de fichiers chaud, pas d'un conteneur fraîchement
démarré.

## Taille des lots

Le plafond de service est configurable par `MAX_BATCH_REQUESTS` et vaut
désormais **2048** au lieu des 512 codés en dur. Côté site,
`ML_INFERENCE_BATCH_SIZE` fixe la taille réellement envoyée.

Les deux doivent bouger ensemble : un lot plus grand que le plafond du service
est **refusé avec un 422, jamais découpé**. Régler le service d'abord, le site
ensuite.

**2048 est la bonne valeur sur GPU, et plus n'est pas mieux** : le débit plafonne
dès 1024 lignes, et 8192 lignes par tranche est mesuré *plus lent* (19 109 contre
20 769 lignes/s) pour quatre fois la VRAM. Le défaut du site reste à **512**,
volontairement : sans endpoint GPU, cette valeur s'applique aussi au conteneur
netcup, où un lot de 2048 prend près d'une seconde même à 6 threads et bien plus
à un seul. Passer le site à 2048 **en même temps** que le branchement du GPU.

À noter, mesuré de bout en bout à travers HTTP : un lot de 2048 prend 193 ms
côté client contre 94 ms d'inférence pure. La sérialisation JSON représente donc
environ la moitié du temps d'une requête une fois le calcul sur GPU. C'est une
raison de plus de ne pas empiler les petits lots, et la limite qu'il faudrait
attaquer ensuite si le débit devenait à nouveau gênant.

## L'option de déploiement recommandée

**Endpoint de type « Load Balancer », alimenté par une image Docker publiée dans
un registre.**

### Pourquoi « Load Balancer » plutôt que l'endpoint à file d'attente

Un endpoint RunPod classique expose `POST /v2/<id>/runsync`, attend une charge
utile enveloppée dans `{"input": {...}}` et renvoie `{"output": {...}}` après un
éventuel passage par `IN_QUEUE`. Le site ne parle pas ce protocole : il appelle
`/health`, `/predict` et `/predict/batch` en HTTP direct. La file imposerait une
couche d'adaptation côté site et priverait le sondage de disponibilité de sa
sémantique.

Un endpoint « Load Balancer » route le HTTP directement vers le conteneur, sur
`https://<endpoint-id>.api.runpod.ai/<chemin>`. Les routes existantes
fonctionnent telles quelles.

Contrepartie : il ne met pas les requêtes en file, il les refuse quand les
workers sont saturés. Le site gère déjà ce cas — `THROTTLE_RETRY_ATTEMPTS`
réessaie les 429/503 avec backoff, mécanisme écrit à l'origine pour la limite
d'admission de Cloud Run.

### Pourquoi une image Docker plutôt qu'un déploiement depuis GitHub

Le déploiement depuis GitHub ne peut pas fonctionner en l'état : `artifacts/` est
dans `.gitignore`, donc `artifacts/matchup-model.pt` n'existe pas sur le distant
et le `COPY` de `Dockerfile.gpu` échouerait au build. Construire et publier
l'image soi-même est déjà la façon dont netcup est alimenté, et garde le
checkpoint hors de Git.

## 1. Construire et publier l'image GPU

```bash
sha256sum artifacts/matchup-model.pt

docker build -f Dockerfile.gpu -t <votre-registre>/rigged-model-gpu:latest .
docker push <votre-registre>/rigged-model-gpu:latest
```

Le checkpoint est **baké dans l'image**, comme pour le CPU : changer de modèle
veut dire rebuild + push + redéploiement, jamais un simple redémarrage.

Vérification locale, si un GPU NVIDIA est disponible :

```bash
docker compose -f docker-compose.gpu.yml up --build
curl localhost:8080/health    # doit renvoyer "device": "cuda:0"
curl -i localhost:8080/ping   # doit renvoyer 200 sans corps
```

## 2. Créer l'endpoint RunPod

**Serverless → New Endpoint → Deploy from a Docker image**, puis :

- **Endpoint type** : `Load Balancer`.
- **Container image** : `<votre-registre>/rigged-model-gpu:latest`.
- **FlashBoot** : activé.
- **GPU** : le tier le moins cher disponible. Le modèle occupe une centaine de
  mégaoctets de VRAM ; la marge au-delà d'un GPU d'entrée de gamme est payée sans
  être utilisée. Ce qui compte pour ce modèle, c'est la latence de lancement des
  noyaux, pas la puissance de calcul brute.
- **Exposed HTTP port** : `80` — la valeur que RunPod passe dans `$PORT`, sur
  lequel l'image écoute. Laisser `PORT_HEALTH` non défini : le conteneur répond
  au sondage sur le même port que le trafic.
- **Health check path** : `/ping`. Déjà le défaut RunPod ; la route existe dans
  `serve.py` et renvoie 200 quand le modèle est chargé, 204 sinon.
- **Workers** : `min 0` / `max 1` pour commencer. Passer `min 1` si les
  démarrages à froid restent gênants malgré FlashBoot.
- **Idle timeout** : au moins la durée d'un balayage complet.

Variables d'environnement de l'endpoint :

| Variable | Valeur | Rôle |
| --- | --- | --- |
| `MODEL_DEVICE` | `auto` | Prend le GPU s'il est visible, retombe sur CPU sinon. Mettre `cuda` pour rendre l'absence de GPU bruyante. |
| `MODEL_WARMUP` | `1` | Lot à vide au démarrage, pour que le contexte CUDA soit payé avant le trafic. |
| `MAX_BATCH_REQUESTS` | `2048` | Plafond d'un `/predict/batch`. Doit être ≥ `ML_INFERENCE_BATCH_SIZE` du site. |
| `MODEL_NAME` | `symmetric-matchup` | Nom renvoyé par `/health` et `/predict`. |
| `PREDICT_API_KEY` | *(optionnel)* | Secret partagé vérifié par le conteneur, en plus de l'authentification RunPod. |
| `WEB_CONCURRENCY` | `1` | Processus uvicorn. À monter avec la concurrence autorisée par worker. |

## 3. Brancher le site

Dans l'environnement du site, sur netcup :

```bash
# Inchangé : prédictions unitaires et /health restent sur le CPU toujours chaud.
ML_INFERENCE_URL=https://model.riggedroyale.com

# Nouveau : les balayages vectorisés partent sur le GPU.
ML_BATCH_INFERENCE_URL=https://<endpoint-id>.api.runpod.ai
ML_BATCH_INFERENCE_BEARER_TOKEN=<clé API RunPod>

# À monter seulement après avoir monté MAX_BATCH_REQUESTS côté service.
ML_INFERENCE_BATCH_SIZE=2048
```

RunPod authentifie chaque appel avec un en-tête `Authorization: Bearer` ; sans
lui la requête n'atteint jamais le conteneur. Le client l'envoie sur `/health`
comme sur `/predict/batch` — le sondage de disponibilité devait être authentifié
lui aussi, sinon un 401 y ferait perdre le support des formes Hero pendant toute
la durée du cache.

`ML_INFERENCE_BEARER_TOKEN` (sans `_BATCH_`) existe aussi, pour le cas où les
deux déploiements passeraient par RunPod. Le déploiement netcup n'a besoin
d'aucun jeton.

**Point de vigilance** : les deux déploiements doivent servir le **même
checkpoint**. Le site interroge maintenant les capacités de chacun séparément et
adapte chaque requête au déploiement qui va y répondre, mais deux checkpoints
différents produiraient malgré tout des scores incohérents entre le chemin
unitaire et le chemin batch. Comparer les `model_version` des deux `/health`
après chaque déploiement.

## 4. Vérifier

```bash
curl -s https://<endpoint-id>.api.runpod.ai/health \
  -H "Authorization: Bearer $RUNPOD_API_KEY" | jq

curl -s -X POST https://<endpoint-id>.api.runpod.ai/predict \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d @example-matchup.json | jq '.win_probability, .model_version'
```

À contrôler :

- `"device"` vaut `cuda:0`. S'il indique `cpu`, l'endpoint a démarré sans GPU
  visible et sert quand même — c'est le repli volontaire de `resolve_device`, et
  c'est ce que ce champ sert à rendre visible.
- `model_version` identique à celui de `model.riggedroyale.com`
  (`v<feature_version>-<sha256[:12] du .pt>`).
- `max_batch_requests` cohérent avec `ML_INFERENCE_BATCH_SIZE`.

### Écart numérique CPU / GPU : mesuré, et à connaître

Le même checkpoint ne donne **pas** exactement le même nombre sur CPU et sur
GPU. Sur 20 000 matchups comparés ligne à ligne :

| Mesure | Valeur |
| --- | --- |
| Écart maximal de probabilité | 6,6 · 10⁻⁴ |
| Écart moyen | 6,4 · 10⁻⁵ |
| p99 | 2,8 · 10⁻⁴ |
| `matchup_label` différent | 0,040 % des matchups |
| `confidence` différente | 0,090 % |
| Pourcentage affiché différent | 0,73 % |

C'est de l'ordre d'accumulation float32 : les noyaux CUDA somment dans un ordre
différent des noyaux CPU, et un transformeur amplifie l'écart. Ce n'est pas TF32
— vérifié, `matmul.allow_tf32` est déjà à `False` et le désactiver explicitement
ne change strictement rien. Ce n'est donc pas corrigeable par configuration, et
ce n'est pas un bug.

Conséquence à assumer pour l'architecture hybride : un même matchup peut
s'afficher à 54 % par le chemin unitaire et 55 % par le chemin batch, dans
environ sept cas sur mille. Un panel donné reste cohérent avec lui-même, puisque
tout un balayage passe par le même device ; l'incohérence n'existe qu'entre les
deux chemins.

Un écart **supérieur à 10⁻³** sur un même matchup, en revanche, sort de ce cadre
et signale autre chose : checkpoint différent, ou précision réduite activée
quelque part.

## Revenir en arrière

Le service netcup n'a pas été touché. Vider `ML_BATCH_INFERENCE_URL` renvoie les
balayages sur le CPU et rétablit exactement le comportement précédent, sans
redéploiement du site autre que la variable d'environnement.
