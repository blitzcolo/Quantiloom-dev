#!/bin/bash
# ============================================================================
# Quantiloom - Multiband Fusion Example Script
# ============================================================================
# Renders VIS/SWIR/MWIR bands and fuses them into a single enhanced image
#
# Usage:
#   ./scripts/multiband_fusion_example.sh <config.toml>
#
# Example:
#   ./scripts/multiband_fusion_example.sh assets/configs/gltf_pbr_test.toml
# ============================================================================

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

if [ $# -ne 1 ]; then
    echo "Usage: $0 <config.toml>"
    echo ""
    echo "Example:"
    echo "  $0 assets/configs/gltf_pbr_test.toml"
    exit 1
fi

CONFIG_FILE="$1"
OUTPUT_DIR="output/fusion"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "  Quantiloom - Multiband Fusion"
echo "========================================"
echo "Config: $CONFIG_FILE"
echo "Output: $OUTPUT_DIR"
echo ""

# ============================================================================
# Step 1: Render VIS band (RGB mode for fast preview, or vis_fused for spectral)
# ============================================================================

echo "[1/4] Rendering VIS band (rgb mode)..."

# Create temporary config for VIS
VIS_CONFIG="${OUTPUT_DIR}/vis_config.toml"
cp "$CONFIG_FILE" "$VIS_CONFIG"

# Modify spectral mode to rgb (fast mode)
sed -i 's/^mode = .*/mode = "rgb"/' "$VIS_CONFIG"
# Modify output path
sed -i "s|^output = .*|output = \"${OUTPUT_DIR}/vis_output.exr\"|" "$VIS_CONFIG"

# Run renderer
./build/src/app/Quantiloom "$VIS_CONFIG"

echo "  VIS rendering complete: ${OUTPUT_DIR}/vis_output.exr"
echo ""

# ============================================================================
# Step 2: Render SWIR band
# ============================================================================

echo "[2/4] Rendering SWIR band (SWIR_Fused)..."

# Create temporary config for SWIR
SWIR_CONFIG="${OUTPUT_DIR}/swir_config.toml"
cp "$CONFIG_FILE" "$SWIR_CONFIG"

# Modify spectral mode to swir_fused
sed -i 's/^mode = .*/mode = "swir_fused"/' "$SWIR_CONFIG"
# Modify output path
sed -i "s|^output = .*|output = \"${OUTPUT_DIR}/swir_output.exr\"|" "$SWIR_CONFIG"

# Run renderer
./build/src/app/Quantiloom "$SWIR_CONFIG"

echo "  SWIR rendering complete: ${OUTPUT_DIR}/swir_output.exr"
echo ""

# ============================================================================
# Step 3: Render MWIR band
# ============================================================================

echo "[3/4] Rendering MWIR band (MWIR_Fused)..."

# Create temporary config for MWIR
MWIR_CONFIG="${OUTPUT_DIR}/mwir_config.toml"
cp "$CONFIG_FILE" "$MWIR_CONFIG"

# Modify spectral mode to mwir_fused
sed -i 's/^mode = .*/mode = "mwir_fused"/' "$MWIR_CONFIG"
# Modify output path
sed -i "s|^output = .*|output = \"${OUTPUT_DIR}/mwir_output.exr\"|" "$MWIR_CONFIG"

# Run renderer
./build/src/app/Quantiloom "$MWIR_CONFIG"

echo "  MWIR rendering complete: ${OUTPUT_DIR}/mwir_output.exr"
echo ""

# ============================================================================
# Step 4: Fuse bands
# ============================================================================

echo "[4/4] Fusing VIS/SWIR/MWIR bands..."

./build/src/tools/fusion_tool \
    "$CONFIG_FILE" \
    "${OUTPUT_DIR}/vis_output.exr" \
    "${OUTPUT_DIR}/swir_output.exr" \
    "${OUTPUT_DIR}/mwir_output.exr" \
    "${OUTPUT_DIR}/fused_output.exr"

echo ""
echo "========================================"
echo "  Multiband Fusion COMPLETE"
echo "========================================"
echo "Outputs:"
echo "  VIS:   ${OUTPUT_DIR}/vis_output.exr"
echo "  SWIR:  ${OUTPUT_DIR}/swir_output.exr"
echo "  MWIR:  ${OUTPUT_DIR}/mwir_output.exr"
echo "  Fused: ${OUTPUT_DIR}/fused_output.exr"
echo "  PNG:   ${OUTPUT_DIR}/fused_output.png"
echo "========================================"
