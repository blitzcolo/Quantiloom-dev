#!/bin/bash
# Deploy generated docs to a separate public repository
# Usage: ./scripts/deploy_docs.sh <docs-repo-url>
# Example: ./scripts/deploy_docs.sh git@github.com:yourname/quantiloom-docs.git

set -e

DOCS_REPO="$1"

if [[ -z "$DOCS_REPO" ]]; then
    echo "Usage: $0 <docs-repo-url>"
    echo "Example: $0 git@github.com:yourname/quantiloom-docs.git"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DOCS_HTML="$PROJECT_ROOT/docs/html"

if [[ ! -d "$DOCS_HTML" ]]; then
    echo "Error: docs/html not found. Run ./scripts/generate_docs.sh first."
    exit 1
fi

TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

echo "Cloning docs repo..."
git clone --depth 1 "$DOCS_REPO" "$TEMP_DIR"

echo "Copying generated docs..."
rm -rf "$TEMP_DIR"/*
cp -r "$DOCS_HTML"/* "$TEMP_DIR/"

# Add .nojekyll for GitHub Pages
touch "$TEMP_DIR/.nojekyll"

cd "$TEMP_DIR"
git add -A
git commit -m "docs: update documentation $(date +%Y-%m-%d)" || echo "No changes to commit"
git push

echo "Documentation deployed to $DOCS_REPO"
