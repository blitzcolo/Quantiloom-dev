#!/usr/bin/env bash
# What is left of the Vulkan coupling outside gpu/vulkan/ (P05 evidence tool).
#
#   bash scripts/gpu-ab/vk_residual.sh
#
# Prints, ASCII only:
#   1. T1, the raw-call residual every P05 step reports: each `vk*(` in
#      src/libQuantiloom outside gpu/vulkan/ (323 before the wave, 0 after S19);
#   2. the same hits per file;
#   3. T2 per orchestration file of design p05-rt-rhi.md section 3: the lines that
#      hold a Vk*/VK_*/VMA_*/Vma* token (the design's `grep -cE`), and how many
#      lines include vulkan.h. A file a later step creates, or D1 deletes, prints
#      "absent" rather than 0;
#   4. whether gpu/vulkan/VulkanBridge.* still exists (T2 wants it gone).
# Read-only: it greps the working tree and changes nothing.
set -euo pipefail
cd "$(dirname "$0")/../.."

LIB=src/libQuantiloom
T1_PATTERN='\bvk[A-Z][A-Za-z0-9]+\('
T2_PATTERN='\bVk[A-Z]\w+|\bVK_[A-Z_0-9]+|\bVMA_|Vma[A-Z]'
INCLUDE_PATTERN='#[[:space:]]*include[[:space:]]*[<"]vulkan/vulkan(_core)?\.h[>"]'

# The files section 3 names, their headers, the ones that only carry Vulkan
# types (S15/S16/S19), the files S9/S14/S6 create, and the dead files of D1.
T2_FILES=(
    renderer/ExternalRenderContext.cpp
    renderer/OfflineRenderer.cpp
    renderer/RenderCore.hpp
    renderer/RenderCore.cpp
    renderer/RayTracingPipeline.hpp
    renderer/RayTracingPipeline.cpp
    renderer/RtBindings.hpp
    renderer/AccelerationStructure.hpp
    renderer/AccelerationStructure.cpp
    renderer/TextureManager.hpp
    renderer/TextureManager.cpp
    renderer/TextureTransitions.hpp
    renderer/TextureTransitions.cpp
    renderer/GpuAlloc.hpp
    renderer/ThermalExchangePrecompute.hpp
    renderer/ThermalExchangePrecompute.cpp
    renderer/PerformanceLogger.hpp
    renderer/PerformanceLogger.cpp
    renderer/AsyncPixelReadback.hpp
    renderer/AsyncPixelReadback.cpp
    renderer/DeferredBufferUploads.hpp
    renderer/DeferredBufferUploads.cpp
    renderer/ThermalPreview.hpp
    renderer/ThermalPreview.cpp
    renderer/ThermalEpochBuilder.hpp
    renderer/ThermalEpochBuilder.cpp
    renderer/RenderDeviceImpl.hpp
    renderer/RenderDevice.cpp
    renderer/TextureCompressor.hpp
    hs_core/BatchRenderer.hpp
    hs_core/BatchRenderer.cpp
    hs_core/HyperspectralRenderer.cpp
    renderer/Upload.hpp
    renderer/Upload.cpp
    renderer/EnvironmentPrefilter.hpp
    core/PerformanceMetrics.hpp
)

t1_hits() {
    { grep -rEo "$T1_PATTERN" "$LIB" --include=*.cpp --include=*.hpp --include=*.h || true; } \
        | { grep -v "^$LIB/gpu/vulkan/" || true; }
}

hits="$(t1_hits)"
if [[ -z "$hits" ]]; then
    total=0
else
    total="$(printf '%s\n' "$hits" | wc -l | tr -d ' ')"
fi

echo "T1: raw vk*( calls in $LIB outside gpu/vulkan/"
echo "  grep -rEo '$T1_PATTERN' $LIB --include=*.cpp --include=*.hpp --include=*.h | grep -v '^$LIB/gpu/vulkan/' | wc -l"
echo "  total: $total"
echo
echo "T1 per file:"
if [[ "$total" -eq 0 ]]; then
    echo "  (none)"
else
    printf '%s\n' "$hits" | cut -d: -f1 | sort | uniq -c | sort -k1,1nr -k2,2 \
        | awk '{ printf "  %5d  %s\n", $1, $2 }'
fi
echo
echo "T2: lines with a Vk*/VK_*/VMA_*/Vma* token, and vulkan.h includes"
echo "  grep -cE '$T2_PATTERN' <file>"
printf '  %-46s %6s %9s\n' "file" "T2" "vulkan.h"
t2_total=0
for file in "${T2_FILES[@]}"; do
    path="$LIB/$file"
    if [[ ! -f "$path" ]]; then
        printf '  %-46s %6s %9s\n' "$file" "absent" "-"
        continue
    fi
    count="$(grep -cE "$T2_PATTERN" "$path" || true)"
    includes="$(grep -cE "$INCLUDE_PATTERN" "$path" || true)"
    t2_total=$((t2_total + count))
    printf '  %-46s %6d %9d\n' "$file" "$count" "$includes"
done
echo "  T2 total over present files: $t2_total"
echo
bridge="$(ls "$LIB"/gpu/vulkan/VulkanBridge.* 2>/dev/null || true)"
if [[ -z "$bridge" ]]; then
    echo "VulkanBridge: gone"
else
    echo "VulkanBridge: present"
    printf '%s\n' "$bridge" | sed 's/^/  /'
fi
