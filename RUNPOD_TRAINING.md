# RunPod training guide

This repo can train on RunPod with one single GPU. The current trainer does not
use DDP/DataParallel, so do not pay for 2 GPUs unless you add multi-GPU support.

## Best cheap config

For the current prepared dataset (~21M battles total, ~2.1 GB Parquet prepared)
and the small model (~0.55M parameters), pick:

- Best value: 1x RTX 3090 Community Cloud, 24 GB VRAM, ideally >=8 vCPU and
  >=40 GB RAM, 60-100 GB volume disk.
- Faster if still cheap: 1x RTX 4090 Community Cloud, 24 GB VRAM, same storage.
- Cheapest smoke/full test: 1x RTX A5000 24 GB. It should fit; it may be slower.

Avoid A100/H100 for this project unless you later increase the architecture a
lot. Avoid multi-GPU for now: the code will mostly use GPU 0 only.

Suggested RunPod settings:

- Template: official RunPod PyTorch template with CUDA, JupyterLab, and SSH.
- GPU count: 1.
- Container disk: 30-40 GB.
- Volume disk: 60-100 GB for Community Cloud.
- Network volume: only if you use Secure Cloud and want permanent reusable data.

## Kaggle in parallel

Use Kaggle for a free smoke run or a baseline run. Be aware that `GPU T4 x2`
does not help much here unless the code is changed to use both GPUs. The current
trainer uses one CUDA device.

Recommended split:

- Kaggle: quick sanity run, fewer epochs if needed.
- RunPod: full run on the same prepared dataset.

## Data options

Option A is cleanest if your raw shards are already in Supabase Storage:

```bash
git clone https://github.com/mael-guimoyas/rigged-royale-matchup-ml.git
cd rigged-royale-matchup-ml

export SUPABASE_URL="https://YOURREF.supabase.co"
export SUPABASE_SECRET_KEY="sb_secret_or_service_role"

bash scripts/runpod_train.sh
```

The script will pull Storage shards, rebuild `prepare`, pretrain card embeddings,
train, evaluate, and benchmark.

Option B uses your local prepared dataset:

```powershell
# Local Windows PowerShell, after installing runpodctl:
runpodctl send data\prepared
```

On the Pod:

```bash
cd /workspace
runpodctl receive YOUR-CODE-FROM-SEND
mkdir -p /workspace/data
mv prepared /workspace/data/prepared

git clone https://github.com/mael-guimoyas/rigged-royale-matchup-ml.git
cd rigged-royale-matchup-ml
bash scripts/runpod_train.sh
```

For repeated or large transfers, use `rsync` over SSH instead of `runpodctl`.

## Tunables

Defaults in `scripts/runpod_train.sh`:

```bash
RUNPOD_BATCH_SIZE=4096
RUNPOD_EVAL_BATCH_SIZE=8192
RUNPOD_NUM_WORKERS=4
RUNPOD_EPOCHS=10
```

If you hit CUDA OOM, retry with:

```bash
RUNPOD_BATCH_SIZE=2048 RUNPOD_EVAL_BATCH_SIZE=4096 bash scripts/runpod_train.sh
```

If GPU usage is low and CPU/RAM are comfortable, try:

```bash
RUNPOD_BATCH_SIZE=8192 RUNPOD_EVAL_BATCH_SIZE=16384 RUNPOD_NUM_WORKERS=8 bash scripts/runpod_train.sh
```

## After training

Artifacts are written to:

```text
/workspace/artifacts
```

Send them back before terminating the Pod:

```bash
runpodctl send /workspace/artifacts
```

Then receive locally with the command printed by RunPod.

Stop billing by stopping or terminating the Pod when finished. If you used a
volume disk, copy the artifacts first because the disk is deleted when the Pod
is terminated. If you used a network volume, storage keeps billing while it
exists.
