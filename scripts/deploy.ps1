# Redeploy the matchup model to Cloud Run.
# Usage:  pwsh scripts/deploy.ps1
# Updating the model = replace artifacts/matchup-model.pt, then run this.
# The PREDICT_API_KEY secret and MODEL_NAME env persist across redeploys, so
# this only rebuilds the image (with the new checkpoint baked in) and rolls it out.
$ErrorActionPreference = "Stop"

$Project = "project-94830673-9a6c-4635-a70"
$Region  = "europe-west1"
$Service = "rigged-matchup"

# gcloud (winget install) is under LocalAppData; add to PATH for this session.
$env:Path = "$env:LOCALAPPDATA\Google\Cloud SDK\google-cloud-sdk\bin;$env:Path"

# Run from the repo root so --source picks up Dockerfile + src + the checkpoint.
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
  gcloud run deploy $Service `
    --source . `
    --region $Region `
    --port 8080 `
    --cpu 1 --memory 512Mi `
    --min-instances 0 `
    --allow-unauthenticated `
    --set-env-vars MODEL_NAME=symmetric-matchup `
    --set-secrets PREDICT_API_KEY=rigged-matchup-key:latest `
    --project $Project
} finally {
  Pop-Location
}
