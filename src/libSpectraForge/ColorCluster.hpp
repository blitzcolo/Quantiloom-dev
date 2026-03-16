/**
 * @file ColorCluster.hpp
 * @brief Texture color analysis via K-means clustering in CIELAB space
 *
 * Clusters texture pixels by perceptual color to identify distinct surface
 * regions. Each cluster centroid maps to an IR material class.
 */

#pragma once

#include "Platform.hpp"
#include "core/Types.hpp"

namespace spectraforge {

using quantiloom::f32;
using quantiloom::u8;
using quantiloom::u32;
using quantiloom::Vector;

struct LABColor {
    f32 L = 0.0f;  // Lightness [0, 100]
    f32 a = 0.0f;  // Green-red [-128, 127]
    f32 b = 0.0f;  // Blue-yellow [-128, 127]
};

struct SF_API ClusterResult {
    Vector<LABColor> centroids;       // K cluster centers
    Vector<u32>      clusterSizes;    // pixel count per cluster
    u32              dominantCluster;  // index of largest cluster
};

// Cluster texture pixels by color in CIELAB space.
// pixels: raw RGBA/RGB data. isSRGB: apply sRGB gamma decode.
SF_API ClusterResult ClusterTextureColors(
    const u8* pixels, u32 width, u32 height, u32 channels,
    bool isSRGB, u32 K, u32 maxIterations = 20
);

// Convert linear RGB [0,1] to CIELAB (D65 illuminant)
SF_API LABColor RGBToLAB(f32 r, f32 g, f32 b);

} // namespace spectraforge
