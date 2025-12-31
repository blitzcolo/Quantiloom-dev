# Deploy generated docs to a separate public repository
# Usage: .\scripts\deploy_docs.ps1 -DocsRepo <repo-url>
# Example: .\scripts\deploy_docs.ps1 -DocsRepo "git@github.com:yourname/quantiloom-docs.git"

param(
    [Parameter(Mandatory=$true)]
    [string]$DocsRepo
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$DocsHtml = Join-Path $ProjectRoot "docs\html"

if (-not (Test-Path $DocsHtml)) {
    Write-Error "Error: docs\html not found. Run .\scripts\generate_docs.ps1 first."
    exit 1
}

$TempDir = Join-Path $env:TEMP "quantiloom-docs-deploy-$(Get-Random)"

try {
    Write-Host "Cloning docs repo..." -ForegroundColor Cyan
    git clone --depth 1 $DocsRepo $TempDir
    if ($LASTEXITCODE -ne 0) { throw "git clone failed" }

    Write-Host "Copying generated docs..." -ForegroundColor Cyan
    Get-ChildItem $TempDir -Exclude ".git" | Remove-Item -Recurse -Force
    Copy-Item -Path "$DocsHtml\*" -Destination $TempDir -Recurse

    # Add .nojekyll for GitHub Pages
    New-Item -Path (Join-Path $TempDir ".nojekyll") -ItemType File -Force | Out-Null

    Push-Location $TempDir
    try {
        git add -A
        $Date = Get-Date -Format "yyyy-MM-dd"
        git commit -m "docs: update documentation $Date"
        if ($LASTEXITCODE -eq 0) {
            git push
            if ($LASTEXITCODE -ne 0) { throw "git push failed" }
        } else {
            Write-Host "No changes to commit" -ForegroundColor Yellow
        }
    }
    finally {
        Pop-Location
    }

    Write-Host "Documentation deployed to $DocsRepo" -ForegroundColor Green
}
finally {
    if (Test-Path $TempDir) {
        Remove-Item $TempDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}
