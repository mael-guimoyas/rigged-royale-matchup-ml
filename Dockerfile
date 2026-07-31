# CPU inference image for the antisymmetric matchup model.
FROM python:3.11-slim

WORKDIR /app

# Install the package + serving extras. Pull the CPU-only torch wheel to keep
# the image small (no CUDA).
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu ".[serve]"

# Bake the trained checkpoint into the image. Cloud Run has no writable volume
# mounts, so the model must ship in the image (it is ~1.6 MB). Local dev can still
# override the path / mount a different checkpoint via MODEL_CHECKPOINT.
COPY artifacts/matchup-model.pt ./artifacts/matchup-model.pt
ENV MODEL_CHECKPOINT=/app/artifacts/matchup-model.pt

# One BLAS/OpenMP thread per request. Measured on Cloud Run (4 vCPU, idle
# revisions of this same image, load driven against each): 1 thread beats the
# uncapped default by ~21% client-side / ~30% server-side p50 under concurrent
# load, while 4 threads is ~10% SLOWER than uncapped. The tensors here are small
# enough that intra-op parallelism costs more in coordination than it returns —
# throughput comes from serving requests concurrently, not from splitting one.
ENV OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

# Cloud Run injects $PORT (defaults to 8080). Use `sh -c` so the variable
# expands, with `exec` so uvicorn becomes PID 1 and receives stop signals.
# EXPOSE is documentation only.
EXPOSE 8080
CMD ["sh", "-c", "exec uvicorn rigged_matchup_ml.serve:app --host 0.0.0.0 --port ${PORT:-8080}"]
