#!/bin/bash
# Generate Doxygen documentation locally
# Usage: ./scripts/generate_docs.sh [--open]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Check dependencies
if ! command -v doxygen &> /dev/null; then
    echo "Error: doxygen not found. Install with:"
    echo "  Ubuntu/Debian: sudo apt install doxygen graphviz"
    echo "  macOS: brew install doxygen graphviz"
    echo "  Windows: choco install doxygen.install graphviz"
    exit 1
fi

if ! command -v dot &> /dev/null; then
    echo "Warning: graphviz (dot) not found. Diagrams will be disabled."
fi

echo "Generating documentation..."
doxygen Doxyfile

echo "Documentation generated at: $PROJECT_ROOT/docs/html/index.html"

# Open in browser if --open flag provided
if [[ "$1" == "--open" ]]; then
    if command -v xdg-open &> /dev/null; then
        xdg-open "$PROJECT_ROOT/docs/html/index.html"
    elif command -v open &> /dev/null; then
        open "$PROJECT_ROOT/docs/html/index.html"
    elif command -v start &> /dev/null; then
        start "$PROJECT_ROOT/docs/html/index.html"
    else
        echo "Cannot detect browser opener. Open manually."
    fi
fi
