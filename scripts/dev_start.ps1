#Requires -Version 5.1
<#
.SYNOPSIS
  Clean stale dev servers and start a single Nutella viewer backend.

.NOTES
  The CAD viewer frontend (demo_viewer.html) is served by serve_viewer.py itself.
  Optional: pass -WithWeb to also start FastAPI (8000) + Vite (5173).
#>
param(
    [int] $ViewerPort = 8765,
    [switch] $WithWeb,
    [switch] $NoBrowser,
    [switch] $Dev
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root

$Python = & (Join-Path $PSScriptRoot "Get-ProjectPython.ps1") -Root $Root
Write-Host "Python interpreter : $Python" -ForegroundColor DarkGray

function Test-OcpAvailable {
    param([string] $Interpreter)
    & $Interpreter -c "import OCP" 2>$null
    return ($LASTEXITCODE -eq 0)
}

if (-not (Test-OcpAvailable -Interpreter $Python)) {
    Write-Host "`nOCP (cadquery-ocp) missing in this Python environment." -ForegroundColor Yellow
    Write-Host "Installing project dependencies: pip install -e `".[dev]`"" -ForegroundColor Yellow
    & $Python -m pip install -e ".[dev]"
    if (-not (Test-OcpAvailable -Interpreter $Python)) {
        throw @"
cadquery-ocp is required for STEP B-Rep import (import OCP).
Install manually with:
  $Python -m pip install -e ".[dev]"
"@
    }
}

function Wait-ViewerReady {
    param(
        [System.Diagnostics.Process] $Process,
        [int] $Port,
        [int] $TimeoutSeconds = 90
    )
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if ($Process.HasExited) {
            throw "Le viewer s'est arrêté avant d'écouter sur le port ${Port} (code $($Process.ExitCode))."
        }
        try {
            $response = Invoke-WebRequest `
                -Uri "http://127.0.0.1:${Port}/api/runtime" `
                -UseBasicParsing `
                -TimeoutSec 2
            if ($response.StatusCode -eq 200) {
                return
            }
        } catch {
            Start-Sleep -Milliseconds 500
        }
    }
    throw "Le viewer PID $($Process.Id) n'est pas prêt après ${TimeoutSeconds} s (port ${Port})."
}

$cleanParams = @{ Force = $true }
if ($ViewerPort -ne 8765) {
    $cleanParams.Ports = @($ViewerPort, 8765, 8766, 8767, 8768, 8769, 8770, 8000, 5173)
}
& (Join-Path $PSScriptRoot "dev_clean.ps1") @cleanParams

Write-Host "`n== Starting viewer backend ==" -ForegroundColor Cyan
$viewerArgs = @("scripts/serve_viewer.py", "--port", "$ViewerPort")
if ($Dev -or $env:NUTELLA_VIEWER_DEV -eq "1") { $viewerArgs += "--dev" }
if ($NoBrowser) { $viewerArgs += "--no-browser" }

$viewer = Start-Process -FilePath $Python -ArgumentList $viewerArgs -WorkingDirectory $Root -PassThru -WindowStyle Normal
Wait-ViewerReady -Process $viewer -Port $ViewerPort

$apiPid = $null
$vitePid = $null

if ($WithWeb) {
    Write-Host "`n== Starting FastAPI backend (8000) ==" -ForegroundColor Cyan
    $api = Start-Process -FilePath $Python -ArgumentList @("-m", "nutella_scraper.cli.main", "serve-api", "--port", "8000") -WorkingDirectory $Root -PassThru -WindowStyle Normal
    $apiPid = $api.Id
    Start-Sleep -Seconds 2

    Write-Host "`n== Starting Vite frontend (5173) ==" -ForegroundColor Cyan
    $vite = Start-Process -FilePath "npm" -ArgumentList @("run", "dev", "--", "--host", "127.0.0.1", "--port", "5173") -WorkingDirectory (Join-Path $Root "web") -PassThru -WindowStyle Normal
    $vitePid = $vite.Id
    Start-Sleep -Seconds 2
}

Write-Host "`n== Dev stack ==" -ForegroundColor Green
Write-Host "Viewer backend PID : $($viewer.Id)  -> http://127.0.0.1:$ViewerPort/index.html"
if ($apiPid) { Write-Host "FastAPI PID        : $apiPid  -> http://127.0.0.1:8000/v1/health" }
if ($vitePid) { Write-Host "Vite frontend PID  : $vitePid  -> http://127.0.0.1:5173/" }

# Write PID file for dev_clean reference
$statePath = Join-Path $Root "output/.dev_state.json"
@{
    started_at = (Get-Date).ToString("o")
    viewer = @{ pid = $viewer.Id; port = $ViewerPort; url = "http://127.0.0.1:$ViewerPort/index.html" }
    api = if ($apiPid) { @{ pid = $apiPid; port = 8000 } } else { $null }
    vite = if ($vitePid) { @{ pid = $vitePid; port = 5173 } } else { $null }
} | ConvertTo-Json -Depth 4 | Set-Content -Path $statePath -Encoding UTF8
Write-Host "State written to output/.dev_state.json"

Start-Sleep -Seconds 1
& (Join-Path $PSScriptRoot "dev_verify.ps1") -ViewerPort $ViewerPort
