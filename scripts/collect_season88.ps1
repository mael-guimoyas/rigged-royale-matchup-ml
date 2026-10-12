[CmdletBinding()]
param(
    [ValidateRange(0.01, 24.0)]
    [double]$DurationHours = 10.0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$config = Join-Path $repoRoot "config\default.yaml"
$seeds = Join-Path $repoRoot "seeds_season88.txt"
$logDir = Join-Path $repoRoot "logs\season-87"

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Python environment not found: $python"
}
if (-not (Test-Path -LiteralPath $seeds -PathType Leaf)) {
    throw "Seed file not found: $seeds"
}

New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$logPath = Join-Path $logDir "collect-$timestamp.log"
$statusPath = Join-Path $logDir "collect-$timestamp.status.log"
$exitCode = 1

$collectorArgs = @(
    "-u", "-m", "rigged_matchup_ml.cli", "collect-api",
    "--config", $config,
    "--tags-file", $seeds,
    "--max-duration-hours", $DurationHours.ToString([Globalization.CultureInfo]::InvariantCulture),
    "--requests-per-second", "30",
    "--workers", "0",
    "--api-token-mode", "1",
    "--upload",
    "--bucket", "training-battles",
    "--prefix", "season-88",
    "--no-progress",
    "--stats-interval", "60"
)

Push-Location $repoRoot
try {
    $process = Start-Process `
        -FilePath $python `
        -ArgumentList $collectorArgs `
        -WorkingDirectory $repoRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $logPath `
        -RedirectStandardError $statusPath `
        -Wait `
        -PassThru
    $exitCode = $process.ExitCode
}
catch {
    $_ | Out-String | Add-Content -LiteralPath $statusPath
    $exitCode = 1
}
finally {
    Pop-Location
}

exit $exitCode
