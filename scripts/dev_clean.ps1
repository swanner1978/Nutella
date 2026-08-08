#Requires -Version 5.1
<#
.SYNOPSIS
  Stop stale Nutella dev servers (viewer, API, Vite).

.DESCRIPTION
  Finds Python/Node listeners on project ports and stops only processes whose
  command line matches this repository (serve_viewer, uvicorn, vite, etc.).
#>
param(
    [int[]] $Ports = @(8765, 8766, 8767, 8768, 8769, 8770, 8000, 5173),
    [switch] $Force
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$RootPattern = [regex]::Escape($Root)

function Test-ProjectProcess {
    param([string] $CommandLine)
    if ([string]::IsNullOrWhiteSpace($CommandLine)) { return $false }
    if ($CommandLine -match $RootPattern) {
        return $true
    }
    return (
        $CommandLine -match "serve_viewer\.py" -or
        $CommandLine -match "nutella_scraper\.api" -or
        $CommandLine -match "nutella_scraper\\cli\\main" -or
        $CommandLine -match "uvicorn" -or
        $CommandLine -match "serve-api" -or
        ($CommandLine -match "vite(\.cmd)?(\s|$)" -and $CommandLine -match "web")
    )
}

Write-Host "== Nutella dev-clean ==" -ForegroundColor Cyan
Write-Host "Repository: $Root"

$maxPasses = 5
$allStopped = @()
for ($pass = 1; $pass -le $maxPasses; $pass++) {
    if ($pass -gt 1) { Write-Host "`n-- dev-clean pass ${pass} --" -ForegroundColor DarkCyan }
    $seenPids = @{}
    $stopped = @()

    foreach ($port in $Ports) {
        $connections = @(Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue)
        if ($connections.Count -eq 0) {
            if ($pass -eq 1) { Write-Host "[port ${port}] free" }
            continue
        }

        $portPids = $connections | Select-Object -ExpandProperty OwningProcess -Unique
        foreach ($procId in $portPids) {
            if ($seenPids.ContainsKey($procId)) { continue }

            $process = Get-CimInstance Win32_Process -Filter "ProcessId=$procId" -ErrorAction SilentlyContinue
            if (-not $process) { continue }

            $name = $process.Name
            $cmd = $process.CommandLine
            $seenPids[$procId] = $true

            if (-not (Test-ProjectProcess -CommandLine $cmd)) {
                Write-Host "[port ${port}] skip PID ${procId} (${name}) - not a Nutella dev process" -ForegroundColor Yellow
                continue
            }

            Write-Host "[port ${port}] stopping PID ${procId} (${name})" -ForegroundColor DarkYellow
            Write-Host "  $cmd"
            if ($Force) {
                Stop-Process -Id $procId -Force
            } else {
                Stop-Process -Id $procId
            }
            $stopped += [pscustomobject]@{ Port = $port; PID = $procId; Name = $name; CommandLine = $cmd }
            $allStopped += $stopped[-1]
        }
    }

    Start-Sleep -Milliseconds 400
    $anyListening = $false
    foreach ($port in $Ports) {
        $still = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
        if ($still) { $anyListening = $true; break }
    }
    if (-not $anyListening) { break }
}

Write-Host "`n== Port check ==" -ForegroundColor Cyan
foreach ($port in $Ports) {
    $still = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    if ($still) {
        Write-Host "[port ${port}] STILL LISTENING (PIDs: $($still.OwningProcess -join ', '))" -ForegroundColor Red
    } else {
        Write-Host "[port ${port}] free" -ForegroundColor Green
    }
}

if ($allStopped.Count -eq 0) {
    Write-Host "`nNo Nutella dev process was stopped."
} else {
    Write-Host "`nStopped $($allStopped.Count) process(es)."
}
