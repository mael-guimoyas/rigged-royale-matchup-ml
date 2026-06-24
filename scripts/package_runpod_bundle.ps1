param(
  [string]$OutputPath = "artifacts\runpod-bundle.zip"
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$outputFullPath = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $OutputPath))
$outputDir = Split-Path -Parent $outputFullPath
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

$stageRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("runpod-bundle-" + [guid]::NewGuid())
$stageRepo = Join-Path $stageRoot "rigged-royale-matchup-ml"
New-Item -ItemType Directory -Force -Path $stageRepo | Out-Null

try {
  $files = @(
    "pyproject.toml",
    "README.md",
    "RUNPOD_TRAINING.md",
    ".env.example"
  )
  foreach ($file in $files) {
    $source = Join-Path $repoRoot $file
    if (Test-Path $source) {
      Copy-Item -LiteralPath $source -Destination (Join-Path $stageRepo $file)
    }
  }

  $directories = @(
    "config",
    "scripts",
    "src",
    "tests",
    "data\prepared"
  )
  foreach ($directory in $directories) {
    $source = Join-Path $repoRoot $directory
    if (Test-Path $source) {
      $destination = Join-Path $stageRepo $directory
      New-Item -ItemType Directory -Force -Path (Split-Path -Parent $destination) | Out-Null
      Copy-Item -LiteralPath $source -Destination $destination -Recurse
    }
  }

  Get-ChildItem -Path $stageRepo -Recurse -Directory -Force |
    Where-Object { $_.Name -in @("__pycache__", ".pytest_cache", ".ruff_cache") } |
    Remove-Item -Recurse -Force

  if (Test-Path $outputFullPath) {
    Remove-Item -LiteralPath $outputFullPath -Force
  }
  Compress-Archive -Path (Join-Path $stageRoot "*") -DestinationPath $outputFullPath -CompressionLevel Optimal

  $sizeGb = [math]::Round((Get-Item -LiteralPath $outputFullPath).Length / 1GB, 2)
  Write-Host "Created $outputFullPath ($sizeGb GB)"
}
finally {
  if ((Test-Path $stageRoot) -and $stageRoot.StartsWith([System.IO.Path]::GetTempPath())) {
    Remove-Item -LiteralPath $stageRoot -Recurse -Force
  }
}
