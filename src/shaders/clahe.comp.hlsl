/**
 * @file clahe.comp.hlsl
 * @brief The display path: a tone operator, then a palette
 *
 * This is the only tone mapping the viewport has. The blit to the swapchain is
 * a bare format conversion, so an image whose radiance sits at 1e-2 -- every
 * LWIR render -- is black on screen until something here maps it into [0,1].
 *
 * Two stages, deliberately independent:
 *
 *   1. TONE: a scalar to [0,1]. This is the contrast stretch.
 *   2. PALETTE: that scalar to a colour. This changes no contrast at all.
 *
 * The three tone operators differ in one property that matters more than
 * sharpness for a thermogram -- whether the mapping is the same everywhere in
 * the image:
 *
 *   TONE_LINEAR    stretch [inputMin, inputMax] (a percentile window computed
 *                  on the host). Globally monotone: brighter is hotter,
 *                  everywhere. What a thermal camera calls linear AGC. Passes
 *                  1 and 2 are skipped for it entirely.
 *   TONE_EQUALIZE  histogram equalisation over the WHOLE image, clipped at
 *                  clipLimit. Also globally monotone; more contrast where the
 *                  pixels actually are. Thermal cameras call this plateau AGC.
 *                  Shares the three passes below -- pass 2 simply sums every
 *                  tile's histogram, so all tiles end up with one CDF.
 *   TONE_CLAHE     per-tile equalisation, bilinearly blended. The best local
 *                  detail and the worst radiometry: two pixels at the same
 *                  temperature in different tiles map to different greys, so
 *                  "brighter is hotter" stops holding. Good for finding an
 *                  edge, wrong for reading a temperature off the screen.
 *
 * Three passes, selected by #define at compile time:
 *   Pass 1 (CLAHE_PASS_HISTOGRAM): Build per-tile histograms
 *   Pass 2 (CLAHE_PASS_CDF): Clip, redistribute, compute CDFs
 *   Pass 3 (CLAHE_PASS_APPLY): Apply the mapping, then the palette
 *
 * @author blitzcolo
 */

// Define which pass to compile (set via specialization constant or #define)
// CLAHE_PASS_HISTOGRAM = 0
// CLAHE_PASS_CDF = 1
// CLAHE_PASS_APPLY = 2

// ============================================================================
// Common Definitions
// ============================================================================

#define HISTOGRAM_BINS 256
#define MAX_TILES_X 64
#define MAX_TILES_Y 64

// Tone operators. See the file header for what separates them.
#define TONE_LINEAR   0
#define TONE_EQUALIZE 1
#define TONE_CLAHE    2

// Palettes. Grey is the identity and the only one that leaves a colour image
// alone; everything else replaces the colour with the scalar's own.
#define PALETTE_GREY          0
#define PALETTE_GREY_INVERTED 1
#define PALETTE_IRONBOW       2
#define PALETTE_RAINBOW       3
#define PALETTE_VIRIDIS       4

// Push constants for all passes
struct CLAHEPushConstants {
    uint imageWidth;
    uint imageHeight;
    uint tileCountX;
    uint tileCountY;
    float clipLimit;           // Typically 2.0-4.0
    uint luminanceOnly;        // 1 = process luminance only, 0 = all channels
    float inputMin;            // Pre-computed min value for normalization
    float inputMax;            // Pre-computed max value for normalization
    uint passIndex;            // Which pass is executing (0, 1, or 2)
    uint toneMode;             // TONE_*
    uint palette;              // PALETTE_*
    uint padding;
};

[[vk::push_constant]] CLAHEPushConstants params;

// Input image (R32G32B32A32_SFLOAT) - raw HDR output from ray tracer
[[vk::binding(0, 0)]] RWTexture2D<float4> inputImage;

// Output image (R32G32B32A32_SFLOAT) - CLAHE processed display image
[[vk::binding(1, 0)]] RWTexture2D<float4> outputImage;

// Histogram buffer: tileCountX * tileCountY * HISTOGRAM_BINS * uint
// Layout: histogram[tileY][tileX][bin]
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> histogramBuffer;

// CDF buffer: same layout as histogram, stores cumulative distribution
[[vk::binding(3, 0)]] RWStructuredBuffer<float> cdfBuffer;

// Tile min/max buffer for HDR normalization: tileCountX * tileCountY * 2
[[vk::binding(4, 0)]] RWStructuredBuffer<float> tileMinMaxBuffer;

// ============================================================================
// Helper Functions
// ============================================================================

// BT.709 RGB to luminance
float RGBToLuminance(float3 rgb) {
    return dot(rgb, float3(0.2126f, 0.7152f, 0.0722f));
}

// Get histogram buffer index
uint GetHistogramIndex(uint tileX, uint tileY, uint bin) {
    return (tileY * params.tileCountX + tileX) * HISTOGRAM_BINS + bin;
}

// Get tile min/max buffer index
uint GetMinMaxIndex(uint tileX, uint tileY) {
    return (tileY * params.tileCountX + tileX) * 2;
}

// Normalize value to [0, 1] range
float NormalizeValue(float v, float minVal, float maxVal) {
    if (maxVal <= minVal) return 0.5f;
    return saturate((v - minVal) / (maxVal - minVal));
}

// Map normalized value to histogram bin
uint ValueToBin(float normalizedValue) {
    uint bin = (uint)(normalizedValue * (HISTOGRAM_BINS - 1) + 0.5f);
    return min(bin, HISTOGRAM_BINS - 1);
}

// Get tile index from pixel coordinates
uint2 GetTileIndex(uint2 pixelCoord) {
    uint tileWidth = (params.imageWidth + params.tileCountX - 1) / params.tileCountX;
    uint tileHeight = (params.imageHeight + params.tileCountY - 1) / params.tileCountY;
    return uint2(pixelCoord.x / tileWidth, pixelCoord.y / tileHeight);
}

// ============================================================================
// Palettes
// ============================================================================
// Evenly spaced control points, linearly interpolated. A ramp rather than a
// polynomial fit because the control points are the specification -- someone
// checking a palette against a reference reads numbers, not coefficients.

float3 IronbowPalette(float t) {
    // The classic thermal ramp: black through violet and red into white.
    static const float3 c[8] = {
        float3(0.000f, 0.000f, 0.000f),
        float3(0.110f, 0.020f, 0.260f),
        float3(0.300f, 0.030f, 0.430f),
        float3(0.510f, 0.060f, 0.430f),
        float3(0.730f, 0.170f, 0.310f),
        float3(0.900f, 0.350f, 0.130f),
        float3(0.990f, 0.640f, 0.010f),
        float3(1.000f, 1.000f, 0.850f)
    };
    float x = saturate(t) * 7.0f;
    uint i = min((uint)x, 6u);
    return lerp(c[i], c[i + 1], x - (float)i);
}

float3 RainbowPalette(float t) {
    // Blue through cyan and green to red. High apparent contrast and no
    // perceptual order to speak of, which is exactly why it is not the
    // default -- but it is what "false colour" means to most people.
    static const float3 c[6] = {
        float3(0.0f, 0.0f, 0.5f),
        float3(0.0f, 0.0f, 1.0f),
        float3(0.0f, 1.0f, 1.0f),
        float3(1.0f, 1.0f, 0.0f),
        float3(1.0f, 0.0f, 0.0f),
        float3(0.5f, 0.0f, 0.0f)
    };
    float x = saturate(t) * 5.0f;
    uint i = min((uint)x, 4u);
    return lerp(c[i], c[i + 1], x - (float)i);
}

float3 ViridisPalette(float t) {
    // Perceptually uniform and monotone in lightness, so a difference in
    // colour is a difference in value rather than an artefact of the ramp.
    // The one to reach for when the picture is going into a paper.
    static const float3 c[11] = {
        float3(0.267f, 0.005f, 0.329f),
        float3(0.283f, 0.141f, 0.458f),
        float3(0.254f, 0.265f, 0.530f),
        float3(0.207f, 0.372f, 0.553f),
        float3(0.164f, 0.471f, 0.558f),
        float3(0.128f, 0.567f, 0.551f),
        float3(0.135f, 0.659f, 0.518f),
        float3(0.267f, 0.749f, 0.441f),
        float3(0.478f, 0.821f, 0.318f),
        float3(0.741f, 0.873f, 0.150f),
        float3(0.993f, 0.906f, 0.144f)
    };
    float x = saturate(t) * 10.0f;
    uint i = min((uint)x, 9u);
    return lerp(c[i], c[i + 1], x - (float)i);
}

float3 ApplyPalette(float t, uint palette) {
    if (palette == PALETTE_IRONBOW) return IronbowPalette(t);
    if (palette == PALETTE_RAINBOW) return RainbowPalette(t);
    if (palette == PALETTE_VIRIDIS) return ViridisPalette(t);
    if (palette == PALETTE_GREY_INVERTED) return (1.0f - saturate(t)).xxx;
    return saturate(t).xxx;
}

// Get tile bounds
void GetTileBounds(uint tileX, uint tileY, out uint2 minCoord, out uint2 maxCoord) {
    uint tileWidth = (params.imageWidth + params.tileCountX - 1) / params.tileCountX;
    uint tileHeight = (params.imageHeight + params.tileCountY - 1) / params.tileCountY;

    minCoord = uint2(tileX * tileWidth, tileY * tileHeight);
    maxCoord = uint2(min((tileX + 1) * tileWidth, params.imageWidth),
                     min((tileY + 1) * tileHeight, params.imageHeight));
}

// ============================================================================
// Pass 1: Build Per-Tile Histograms
// ============================================================================
// Each workgroup processes one tile
// Dispatch: (tileCountX, tileCountY, 1)

#ifdef CLAHE_PASS_HISTOGRAM

groupshared uint localHistogram[HISTOGRAM_BINS];

[numthreads(16, 16, 1)]
void main(uint3 groupId : SV_GroupID, uint3 localId : SV_GroupThreadID, uint localIndex : SV_GroupIndex) {
    uint tileX = groupId.x;
    uint tileY = groupId.y;

    // Initialize shared memory histogram to zero
    if (localIndex < HISTOGRAM_BINS) {
        localHistogram[localIndex] = 0;
    }
    GroupMemoryBarrierWithGroupSync();

    // Get tile bounds
    uint2 minCoord, maxCoord;
    GetTileBounds(tileX, tileY, minCoord, maxCoord);
    uint tileWidth = maxCoord.x - minCoord.x;
    uint tileHeight = maxCoord.y - minCoord.y;

    // Use global min/max for normalization (pre-computed on CPU)
    float globalMin = params.inputMin;
    float globalMax = params.inputMax;

    // Each thread processes multiple pixels and builds histogram
    for (uint dy = localId.y; dy < tileHeight; dy += 16) {
        for (uint dx = localId.x; dx < tileWidth; dx += 16) {
            uint2 pixelCoord = minCoord + uint2(dx, dy);
            if (pixelCoord.x >= params.imageWidth || pixelCoord.y >= params.imageHeight)
                continue;

            float4 pixel = inputImage[pixelCoord];

            // Skip invalid pixels
            if (isinf(pixel.r) || isnan(pixel.r) ||
                isinf(pixel.g) || isnan(pixel.g) ||
                isinf(pixel.b) || isnan(pixel.b))
                continue;

            // Compute luminance for histogram
            float value = RGBToLuminance(pixel.rgb);

            // Normalize and bin
            float normalized = NormalizeValue(value, globalMin, globalMax);
            uint bin = ValueToBin(normalized);

            InterlockedAdd(localHistogram[bin], 1);
        }
    }

    GroupMemoryBarrierWithGroupSync();

    // Write histogram to global memory
    if (localIndex < HISTOGRAM_BINS) {
        uint idx = GetHistogramIndex(tileX, tileY, localIndex);
        histogramBuffer[idx] = localHistogram[localIndex];
    }
}

#endif // CLAHE_PASS_HISTOGRAM

// ============================================================================
// Pass 2: Clip, Redistribute, and Compute CDF
// ============================================================================
// Each workgroup processes one tile's histogram
// Dispatch: (tileCountX, tileCountY, 1)

#ifdef CLAHE_PASS_CDF

groupshared uint histogram[HISTOGRAM_BINS];
// The counts as they were loaded. The working copy above is clipped in place,
// and both the pixel total and the clipped excess have to be read from the
// unclipped ones -- this used to be a second pass over global memory.
groupshared uint rawHistogram[HISTOGRAM_BINS];
groupshared float cdf[HISTOGRAM_BINS];

/// One bin's count for this workgroup's tile, or for the whole image when the
/// tone operator is global. Summing every tile here is what makes equalisation
/// global without a second pipeline: every workgroup then computes the same
/// CDF, and pass 3's bilinear blend between four identical CDFs is that CDF.
uint LoadBin(uint tileX, uint tileY, uint bin) {
    if (params.toneMode != TONE_EQUALIZE) {
        return histogramBuffer[GetHistogramIndex(tileX, tileY, bin)];
    }
    uint total = 0;
    for (uint ty = 0; ty < params.tileCountY; ++ty) {
        for (uint tx = 0; tx < params.tileCountX; ++tx) {
            total += histogramBuffer[GetHistogramIndex(tx, ty, bin)];
        }
    }
    return total;
}

[numthreads(256, 1, 1)]
void main(uint3 groupId : SV_GroupID, uint localIndex : SV_GroupIndex) {
    uint tileX = groupId.x;
    uint tileY = groupId.y;

    // Load histogram to shared memory
    if (localIndex < HISTOGRAM_BINS) {
        rawHistogram[localIndex] = LoadBin(tileX, tileY, localIndex);
        histogram[localIndex] = rawHistogram[localIndex];
    }
    GroupMemoryBarrierWithGroupSync();

    // Calculate total pixel count in this tile
    uint totalPixels = 0;
    if (localIndex == 0) {
        for (uint i = 0; i < HISTOGRAM_BINS; ++i) {
            totalPixels += rawHistogram[i];
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // Broadcast totalPixels (stored in histogram[0] temporarily)
    if (localIndex == 0) {
        cdf[0] = (float)totalPixels;
    }
    GroupMemoryBarrierWithGroupSync();
    totalPixels = (uint)cdf[0];

    // Calculate clip limit (absolute count)
    float avgBinCount = (float)totalPixels / (float)HISTOGRAM_BINS;
    uint clipThreshold = (uint)(params.clipLimit * avgBinCount);
    if (clipThreshold == 0) clipThreshold = 1;

    // Clip and count excess
    if (localIndex < HISTOGRAM_BINS) {
        if (histogram[localIndex] > clipThreshold) {
            histogram[localIndex] = clipThreshold;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // Sum excess (simple reduction)
    if (localIndex == 0) {
        uint totalExcess = 0;
        for (uint i = 0; i < HISTOGRAM_BINS; ++i) {
            if (rawHistogram[i] > clipThreshold) {
                totalExcess += rawHistogram[i] - clipThreshold;
            }
        }
        cdf[1] = (float)totalExcess;  // Store for redistribution
    }
    GroupMemoryBarrierWithGroupSync();

    // Redistribute excess equally
    uint totalExcess = (uint)cdf[1];
    uint redistributePerBin = totalExcess / HISTOGRAM_BINS;
    uint remainder = totalExcess % HISTOGRAM_BINS;

    if (localIndex < HISTOGRAM_BINS) {
        histogram[localIndex] += redistributePerBin;
        if (localIndex < remainder) {
            histogram[localIndex] += 1;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // Compute CDF (prefix sum)
    if (localIndex == 0) {
        float cumulative = 0.0f;
        float scale = (totalPixels > 0) ? 1.0f / (float)totalPixels : 1.0f;
        for (uint i = 0; i < HISTOGRAM_BINS; ++i) {
            cumulative += (float)histogram[i];
            cdf[i] = cumulative * scale;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // Write CDF to global memory
    if (localIndex < HISTOGRAM_BINS) {
        uint idx = GetHistogramIndex(tileX, tileY, localIndex);
        cdfBuffer[idx] = cdf[localIndex];
    }
}

#endif // CLAHE_PASS_CDF

// ============================================================================
// Pass 3: Apply Interpolated Mapping
// ============================================================================
// Each thread processes one pixel
// Dispatch: ((width+15)/16, (height+15)/16, 1)

#ifdef CLAHE_PASS_APPLY

// Sample CDF from a tile
float SampleCDF(uint tileX, uint tileY, uint bin) {
    tileX = clamp(tileX, 0, params.tileCountX - 1);
    tileY = clamp(tileY, 0, params.tileCountY - 1);
    bin = clamp(bin, 0, HISTOGRAM_BINS - 1);
    uint idx = GetHistogramIndex(tileX, tileY, bin);
    return cdfBuffer[idx];
}

// Bilinear interpolation of CDF values from 4 surrounding tiles
float InterpolateCDF(float2 pixelCenter, uint bin) {
    float tileWidth = (float)params.imageWidth / (float)params.tileCountX;
    float tileHeight = (float)params.imageHeight / (float)params.tileCountY;

    // Find the four surrounding tile centers
    float tileFracX = (pixelCenter.x / tileWidth) - 0.5f;
    float tileFracY = (pixelCenter.y / tileHeight) - 0.5f;

    int tileX0 = (int)floor(tileFracX);
    int tileY0 = (int)floor(tileFracY);
    int tileX1 = tileX0 + 1;
    int tileY1 = tileY0 + 1;

    float fx = tileFracX - (float)tileX0;
    float fy = tileFracY - (float)tileY0;

    // Clamp tile indices
    tileX0 = clamp(tileX0, 0, (int)params.tileCountX - 1);
    tileY0 = clamp(tileY0, 0, (int)params.tileCountY - 1);
    tileX1 = clamp(tileX1, 0, (int)params.tileCountX - 1);
    tileY1 = clamp(tileY1, 0, (int)params.tileCountY - 1);

    // Sample CDF from 4 tiles
    float cdf00 = SampleCDF(tileX0, tileY0, bin);
    float cdf10 = SampleCDF(tileX1, tileY0, bin);
    float cdf01 = SampleCDF(tileX0, tileY1, bin);
    float cdf11 = SampleCDF(tileX1, tileY1, bin);

    // Bilinear interpolation
    float cdf0 = lerp(cdf00, cdf10, fx);
    float cdf1 = lerp(cdf01, cdf11, fx);
    return lerp(cdf0, cdf1, fy);
}

/// One value to [0,1]. The linear operator is the window and nothing else, so
/// it reads no CDF at all -- which is why the host skips the two passes that
/// would have built one.
float ToneMap(float value, float minValue, float maxValue, float2 pixelCenter) {
    float normalized = NormalizeValue(value, minValue, maxValue);
    if (params.toneMode == TONE_LINEAR) {
        return normalized;
    }
    return InterpolateCDF(pixelCenter, ValueToBin(normalized));
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchId : SV_DispatchThreadID) {
    uint2 pixelCoord = dispatchId.xy;

    if (pixelCoord.x >= params.imageWidth || pixelCoord.y >= params.imageHeight)
        return;

    float4 pixel = inputImage[pixelCoord];
    float4 result = pixel;

    // Skip invalid pixels
    if (isinf(pixel.r) || isnan(pixel.r) ||
        isinf(pixel.g) || isnan(pixel.g) ||
        isinf(pixel.b) || isnan(pixel.b)) {
        outputImage[pixelCoord] = pixel;
        return;
    }

    float globalMin = params.inputMin;
    float globalMax = params.inputMax;
    float2 pixelCenter = float2(pixelCoord) + 0.5f;

    if (params.palette != PALETTE_GREY) {
        // The colour is the palette's, so there is nothing for luminanceOnly to
        // preserve: the scalar is the luminance and the palette decides the
        // rest. Infrared renders are grey to begin with, so nothing is lost
        // there; on a visible render this deliberately discards the colour,
        // which is what asking for false colour means.
        float t = ToneMap(RGBToLuminance(pixel.rgb), globalMin, globalMax, pixelCenter);
        result.rgb = ApplyPalette(t, params.palette);
    } else if (params.luminanceOnly) {
        // Process luminance only, preserve colors
        float luminance = RGBToLuminance(pixel.rgb);
        float normalizedLum = NormalizeValue(luminance, globalMin, globalMax);
        float mappedLum = ToneMap(luminance, globalMin, globalMax, pixelCenter);

        // Scale RGB by luminance ratio
        // Output is normalized to [0, 1] for display (not HDR physical values)
        if (normalizedLum > 1e-6f) {
            float scale = mappedLum / normalizedLum;
            result.rgb = saturate(float3(
                NormalizeValue(pixel.r, globalMin, globalMax) * scale,
                NormalizeValue(pixel.g, globalMin, globalMax) * scale,
                NormalizeValue(pixel.b, globalMin, globalMax) * scale
            ));
        } else {
            result.rgb = float3(mappedLum, mappedLum, mappedLum);
        }
    } else {
        // Process each channel independently
        result.r = ToneMap(pixel.r, globalMin, globalMax, pixelCenter);
        result.g = ToneMap(pixel.g, globalMin, globalMax, pixelCenter);
        result.b = ToneMap(pixel.b, globalMin, globalMax, pixelCenter);
    }

    result.a = pixel.a;
    outputImage[pixelCoord] = result;
}

#endif // CLAHE_PASS_APPLY
