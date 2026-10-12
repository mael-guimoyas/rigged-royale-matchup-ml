[CmdletBinding()]
param(
    [ValidateRange(0.01, 24.0)]
    [double]$DurationHours = 10.0,

    [ValidateRange(2, 31)]
    [int]$Days = 7,

    [string]$TaskName = "RiggedRoyale-Season88-Collect"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$runner = Join-Path $PSScriptRoot "collect_season88.ps1"
$firstScheduledStart = (Get-Date).AddDays(1)
$lastScheduledStart = $firstScheduledStart.AddDays($Days - 2)
$endBoundary = $lastScheduledStart.AddHours(1)
$currentUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name

$actionArgs = @(
    "-NoProfile",
    "-NonInteractive",
    "-WindowStyle", "Hidden",
    "-ExecutionPolicy", "Bypass",
    "-File", ('"{0}"' -f $runner),
    "-DurationHours", $DurationHours.ToString([Globalization.CultureInfo]::InvariantCulture)
) -join " "

$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $actionArgs
$trigger = New-ScheduledTaskTrigger -Daily -At $firstScheduledStart
$trigger.EndBoundary = $endBoundary.ToString("yyyy-MM-dd'T'HH:mm:sszzz")
$principal = New-ScheduledTaskPrincipal `
    -UserId $currentUser `
    -LogonType Interactive `
    -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -WakeToRun `
    -Hidden `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours ($DurationHours + 1))

$task = New-ScheduledTask `
    -Action $action `
    -Trigger $trigger `
    -Principal $principal `
    -Settings $settings `
    -Description "Season 88 battle collection: 10 hours per day, isolated shards and Storage prefix."

Register-ScheduledTask -TaskName $TaskName -InputObject $task -Force | Out-Null
Start-ScheduledTask -TaskName $TaskName

[pscustomobject]@{
    TaskName = $TaskName
    FirstRunStarted = Get-Date
    FirstScheduledStart = $firstScheduledStart
    LastScheduledStart = $lastScheduledStart
    DurationHours = $DurationHours
    TotalDays = $Days
}
