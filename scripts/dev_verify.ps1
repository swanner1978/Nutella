#Requires -Version 5.1
param(
    [int] $ViewerPort = 8765
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$base = "http://127.0.0.1:$ViewerPort"

Write-Host "== Nutella dev-verify (viewer:$ViewerPort) ==" -ForegroundColor Cyan

$index = Invoke-WebRequest -Uri "$base/index.html" -UseBasicParsing
Write-Host "[GET /index.html] HTTP $($index.StatusCode)"
$runtime = Invoke-RestMethod -Uri "$base/api/runtime"
Write-Host "[GET /api/runtime] simulation=$($runtime.simulation_api) status=$($runtime.status_api)"

$verifyJson = python -c @"
import json, urllib.request
req = urllib.request.Request('$base/api/does-not-exist', data=b'{}', method='POST', headers={'Content-Type':'application/json'})
try:
    urllib.request.urlopen(req, timeout=10)
except Exception as exc:
    resp = exc.fp.read().decode() if hasattr(exc, 'fp') and exc.fp else '{}'
    status = exc.code if hasattr(exc, 'code') else 0
    print(json.dumps({'status': status, 'body': json.loads(resp) if resp else {}}))
"@

$result = $verifyJson | ConvertFrom-Json
Write-Host "[POST /api/does-not-exist] HTTP $($result.status)"
Write-Host "  error    : $($result.body.error)"
Write-Host "  message  : $($result.body.message)"

if (-not $result.body.available_endpoints) {
    Write-Host "  WARNING: old server - missing available_endpoints" -ForegroundColor Red
    exit 1
}

Write-Host "  endpoints: $($result.body.available_endpoints -join ', ')" -ForegroundColor Green

$expected = @("/api/import-step", "/api/simulate-contact")
foreach ($endpoint in $expected) {
    if ($result.body.available_endpoints -notcontains $endpoint) {
        Write-Host "  MISSING endpoint: $endpoint" -ForegroundColor Red
        exit 1
    }
}

foreach ($marker in @("simulate-contact", "cancel-simulation", "simulation-progress")) {
    if ($index.Content -notmatch $marker) {
        Write-Host "WARNING: index.html missing ${marker} UI" -ForegroundColor Red
        exit 1
    }
}

if ($runtime.simulation_api -ne "/api/simulate-contact" -or
    $runtime.status_api -ne "/api/simulations/{simulation_id}") {
    Write-Host "WARNING: asynchronous simulation API contract mismatch" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Viewer API contract OK (current code)." -ForegroundColor Green
