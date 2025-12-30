/**
 * @file AdaptiveGridGenerator.cpp
 * @brief Implementation of adaptive wavelength grid generation
 *
 * @author wtflmao
 */

#include "hs_core/AdaptiveGridGenerator.hpp"
#include "core/Log.hpp"

#include <algorithm>
#include <cmath>

namespace quantiloom {

// ============================================================================
// AdaptiveGridGenerator - Grid Generation
// ============================================================================

AdaptiveGridInfo AdaptiveGridGenerator::Generate(
    const HyperspectralConfig& config,
    const SpectralAnalysisResult& analysis
) {
    return Generate(
        config.wavelengthMin_nm,
        config.wavelengthMax_nm,
        config.wavelengthStep_nm,
        analysis.criticalWavelengths,
        config.adaptiveCriticalRadius_nm,
        config.adaptiveCoarseMultiplier
    );
}

AdaptiveGridInfo AdaptiveGridGenerator::Generate(
    f32 wavelengthMin,
    f32 wavelengthMax,
    f32 baseStep,
    const Vector<f32>& criticalWavelengths,
    f32 criticalRadius,
    f32 coarseMultiplier
) {
    AdaptiveGridInfo info;

    // Validate inputs
    if (wavelengthMax <= wavelengthMin || baseStep <= 0.0f) {
        LOG_ERROR("Invalid grid parameters: min={}, max={}, step={}",
                  wavelengthMin, wavelengthMax, baseStep);
        return info;
    }

    // Clamp coarse multiplier to reasonable range
    coarseMultiplier = std::clamp(coarseMultiplier, 1.0f, 8.0f);

    // Sort critical wavelengths for efficient lookup
    Vector<f32> sortedCritical = criticalWavelengths;
    std::sort(sortedCritical.begin(), sortedCritical.end());

    // Generate adaptive grid
    info.wavelengths.reserve(
        static_cast<usize>((wavelengthMax - wavelengthMin) / baseStep) + 1
    );
    info.isCritical.reserve(info.wavelengths.capacity());

    f32 currentWl = wavelengthMin;
    u32 criticalCount = 0;
    u32 coarseCount = 0;

    while (currentWl <= wavelengthMax) {
        info.wavelengths.push_back(currentWl);

        bool isCrit = IsNearCritical(currentWl, sortedCritical, criticalRadius);
        info.isCritical.push_back(isCrit);

        if (isCrit) {
            ++criticalCount;
        } else {
            ++coarseCount;
        }

        // Determine next step size
        // Check if we're approaching a critical region
        f32 distToNext = DistanceToNearestCritical(
            currentWl + baseStep * coarseMultiplier,
            sortedCritical
        );

        if (isCrit || distToNext < criticalRadius) {
            // In or approaching critical region: use fine step
            currentWl += baseStep;
        } else {
            // In flat region: use coarse step
            currentWl += baseStep * coarseMultiplier;
        }

        // Don't overshoot the maximum
        if (currentWl > wavelengthMax && info.wavelengths.back() < wavelengthMax) {
            // Add final wavelength at exactly wavelengthMax
            if (std::abs(info.wavelengths.back() - wavelengthMax) > baseStep * 0.1f) {
                info.wavelengths.push_back(wavelengthMax);
                info.isCritical.push_back(
                    IsNearCritical(wavelengthMax, sortedCritical, criticalRadius)
                );
                if (info.isCritical.back()) ++criticalCount; else ++coarseCount;
            }
            break;
        }
    }

    // Populate metadata
    info.totalWavelengths = static_cast<u32>(info.wavelengths.size());
    info.criticalWavelengths = criticalCount;
    info.coarseWavelengths = coarseCount;

    // Calculate compression ratio
    u32 uniformCount = static_cast<u32>((wavelengthMax - wavelengthMin) / baseStep) + 1;
    info.compressionRatio = static_cast<f32>(uniformCount) /
                            static_cast<f32>(info.totalWavelengths);

    LOG_INFO("Adaptive grid generated: {} wavelengths ({} critical, {} coarse), "
             "compression ratio: {:.2f}x",
             info.totalWavelengths, info.criticalWavelengths,
             info.coarseWavelengths, info.compressionRatio);

    return info;
}

AdaptiveGridInfo AdaptiveGridGenerator::GenerateUniform(
    const HyperspectralConfig& config
) {
    AdaptiveGridInfo info;

    u32 numBands = config.GetNumBands();
    info.wavelengths.reserve(numBands);
    info.isCritical.reserve(numBands);

    for (u32 i = 0; i < numBands; ++i) {
        info.wavelengths.push_back(config.GetWavelength(i));
        info.isCritical.push_back(true);  // All considered critical in uniform mode
    }

    info.totalWavelengths = numBands;
    info.criticalWavelengths = numBands;
    info.coarseWavelengths = 0;
    info.compressionRatio = 1.0f;

    return info;
}

f32 AdaptiveGridGenerator::EstimateCompressionRatio(
    f32 wavelengthMin,
    f32 wavelengthMax,
    f32 baseStep,
    const Vector<f32>& criticalWavelengths,
    f32 criticalRadius,
    f32 coarseMultiplier
) {
    // Calculate uniform count
    u32 uniformCount = static_cast<u32>((wavelengthMax - wavelengthMin) / baseStep) + 1;

    if (criticalWavelengths.empty()) {
        // No critical wavelengths: maximum compression
        u32 coarseCount = static_cast<u32>(
            (wavelengthMax - wavelengthMin) / (baseStep * coarseMultiplier)
        ) + 1;
        return static_cast<f32>(uniformCount) / static_cast<f32>(coarseCount);
    }

    // Estimate based on critical region coverage
    f32 totalRange = wavelengthMax - wavelengthMin;
    f32 criticalCoverage = 0.0f;

    // Calculate non-overlapping critical region extent
    Vector<std::pair<f32, f32>> criticalRegions;
    for (f32 crit : criticalWavelengths) {
        f32 regionMin = std::max(wavelengthMin, crit - criticalRadius);
        f32 regionMax = std::min(wavelengthMax, crit + criticalRadius);

        if (regionMin < regionMax) {
            criticalRegions.emplace_back(regionMin, regionMax);
        }
    }

    // Merge overlapping regions
    if (!criticalRegions.empty()) {
        std::sort(criticalRegions.begin(), criticalRegions.end());
        Vector<std::pair<f32, f32>> merged;
        merged.push_back(criticalRegions[0]);

        for (usize i = 1; i < criticalRegions.size(); ++i) {
            if (criticalRegions[i].first <= merged.back().second) {
                merged.back().second = std::max(merged.back().second,
                                                criticalRegions[i].second);
            } else {
                merged.push_back(criticalRegions[i]);
            }
        }

        for (const auto& region : merged) {
            criticalCoverage += region.second - region.first;
        }
    }

    f32 coarseCoverage = totalRange - criticalCoverage;

    // Estimate sample counts
    f32 criticalSamples = criticalCoverage / baseStep;
    f32 coarseSamples = coarseCoverage / (baseStep * coarseMultiplier);
    f32 adaptiveCount = criticalSamples + coarseSamples;

    return static_cast<f32>(uniformCount) / std::max(1.0f, adaptiveCount);
}

bool AdaptiveGridGenerator::IsNearCritical(
    f32 wavelength,
    const Vector<f32>& criticalWavelengths,
    f32 radius
) {
    for (f32 crit : criticalWavelengths) {
        if (std::abs(wavelength - crit) <= radius) {
            return true;
        }
    }
    return false;
}

f32 AdaptiveGridGenerator::DistanceToNearestCritical(
    f32 wavelength,
    const Vector<f32>& criticalWavelengths
) {
    if (criticalWavelengths.empty()) {
        return std::numeric_limits<f32>::max();
    }

    f32 minDist = std::numeric_limits<f32>::max();
    for (f32 crit : criticalWavelengths) {
        f32 dist = std::abs(wavelength - crit);
        if (dist < minDist) {
            minDist = dist;
        }
    }
    return minDist;
}

// ============================================================================
// Reconstruction Mapping Generation
// ============================================================================

Vector<ReconstructionMapping> GenerateReconstructionMapping(
    const AdaptiveGridInfo& adaptiveGrid,
    const HyperspectralConfig& config
) {
    Vector<ReconstructionMapping> mapping;

    u32 targetBands = config.GetNumBands();
    mapping.reserve(targetBands);

    const auto& srcWavelengths = adaptiveGrid.wavelengths;
    u32 srcBands = static_cast<u32>(srcWavelengths.size());

    if (srcBands == 0) {
        return mapping;
    }

    u32 srcIdx = 0;
    f32 tolerance = config.wavelengthStep_nm * 0.01f;

    for (u32 tgtIdx = 0; tgtIdx < targetBands; ++tgtIdx) {
        f32 targetWl = config.GetWavelength(tgtIdx);

        ReconstructionMapping map;
        map.targetBand = tgtIdx;
        map.targetWavelength = targetWl;

        // Find the source wavelength closest to targetWl
        // Advance srcIdx while the next source wavelength is closer or equal to target
        while (srcIdx < srcBands - 1 && srcWavelengths[srcIdx + 1] <= targetWl + tolerance) {
            ++srcIdx;
        }

        f32 srcWl = srcWavelengths[srcIdx];

        if (std::abs(targetWl - srcWl) < tolerance) {
            // Exact match (or very close) - this wavelength was rendered
            map.wasRendered = true;
            map.sourceBand = srcIdx;
            map.sourceNextBand = srcIdx;
            map.interpWeight = 0.0f;
        } else if (srcIdx < srcBands - 1) {
            // Interpolation needed between srcIdx and srcIdx+1
            f32 srcNextWl = srcWavelengths[srcIdx + 1];

            map.wasRendered = false;
            map.sourceBand = srcIdx;
            map.sourceNextBand = srcIdx + 1;
            map.interpWeight = (targetWl - srcWl) / (srcNextWl - srcWl);
        } else {
            // Beyond last source wavelength - extrapolate from last
            map.wasRendered = false;
            map.sourceBand = srcBands - 1;
            map.sourceNextBand = srcBands - 1;
            map.interpWeight = 0.0f;
        }

        mapping.push_back(map);
    }

    return mapping;
}

} // namespace quantiloom
