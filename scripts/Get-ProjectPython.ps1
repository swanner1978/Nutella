#Requires -Version 5.1
<#
.SYNOPSIS
  Resolve the Python interpreter for Nutella dev scripts.

.DESCRIPTION
  Prefers .venv\Scripts\python.exe in the repository root so dev servers use
  the same environment as pytest and pip install -e ".[dev]".
#>
param(
    [Parameter(Mandatory = $true)]
    [string] $Root
)

$venvPython = Join-Path $Root ".venv\Scripts\python.exe"
if (Test-Path $venvPython) {
    return (Resolve-Path $venvPython).Path
}

Write-Warning @"
Project virtualenv not found at $venvPython
Falling back to 'python' on PATH. Create and install deps with:
  python -m venv .venv
  .venv\Scripts\python.exe -m pip install -e ".[dev]"
"@
return "python"
