# CPU inference image for the antisymmetric matchup model.
FROM python:3.11-slim

WORKDIR /app

# Install the package + serving extras. Pull the CPU-only torch wheel to keep
# the image small (no CUDA).
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu ".[serve]"

# The trained checkpoint is mounted at runtime (see docker-compose.yml) so the
# image stays model-agnostic. Override with MODEL_CHECKPOINT if needed.
ENV MODEL_CHECKPOINT=/app/artifacts/matchup-model.pt

EXPOSE 8080
CMD ["uvicorn", "rigged_matchup_ml.serve:app", "--host", "0.0.0.0", "--port", "8080"]
