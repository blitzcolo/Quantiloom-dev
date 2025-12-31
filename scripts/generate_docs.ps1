# Generate Doxygen documentation locally
# Usage: .\scripts\generate_docs.ps1 [-Open]

param(
    [switch]$Open
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

Push-Location $ProjectRoot

try {
    # Check doxygen
    if (-not (Get-Command doxygen -ErrorAction SilentlyContinue)) {
        Write-Error @"
Error: doxygen not found. Install with:
  choco install doxygen.install graphviz
  or download from https://www.doxygen.nl/download.html
"@
        exit 1
    }

    # Check graphviz
    if (-not (Get-Command dot -ErrorAction SilentlyContinue)) {
        Write-Warning "graphviz (dot) not found. Diagrams will be disabled."
    }

    Write-Host "Generating documentation..." -ForegroundColor Cyan
    doxygen Doxyfile

    if ($LASTEXITCODE -ne 0) {
        Write-Error "Doxygen failed with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }

    $DocsPath = Join-Path $ProjectRoot "docs\html\index.html"
    Write-Host "Documentation generated at: $DocsPath" -ForegroundColor Green

    if ($Open) {
        Start-Process $DocsPath
    }
}
finally {
    Pop-Location
}
