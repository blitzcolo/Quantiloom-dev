#include "ColorCluster.hpp"
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <limits>

namespace spectraforge {

// ============================================================================
// sRGB -> Linear
// ============================================================================

static f32 SRGBToLinear(f32 s) {
    return (s <= 0.04045f) ? s / 12.92f : std::pow((s + 0.055f) / 1.055f, 2.4f);
}

// ============================================================================
// Linear RGB -> XYZ (D65)
// ============================================================================

struct XYZ { f32 x, y, z; };

static XYZ LinearRGBToXYZ(f32 r, f32 g, f32 b) {
    // sRGB to XYZ matrix (D65 illuminant)
    return {
        0.4124564f * r + 0.3575761f * g + 0.1804375f * b,
        0.2126729f * r + 0.7151522f * g + 0.0721750f * b,
        0.0193339f * r + 0.1191920f * g + 0.9503041f * b
    };
}

// ============================================================================
// XYZ -> LAB
// ============================================================================

// D65 reference white
static constexpr f32 REF_X = 0.95047f;
static constexpr f32 REF_Y = 1.00000f;
static constexpr f32 REF_Z = 1.08883f;

static f32 LabF(f32 t) {
    constexpr f32 DELTA = 6.0f / 29.0f;
    constexpr f32 DELTA3 = DELTA * DELTA * DELTA;
    if (t > DELTA3)
        return std::cbrt(t);
    return t / (3.0f * DELTA * DELTA) + 4.0f / 29.0f;
}

static LABColor XYZToLAB(const XYZ& xyz) {
    f32 fx = LabF(xyz.x / REF_X);
    f32 fy = LabF(xyz.y / REF_Y);
    f32 fz = LabF(xyz.z / REF_Z);
    return {
        116.0f * fy - 16.0f,
        500.0f * (fx - fy),
        200.0f * (fy - fz)
    };
}

// ============================================================================
// Public: RGB -> LAB
// ============================================================================

LABColor RGBToLAB(f32 r, f32 g, f32 b) {
    XYZ xyz = LinearRGBToXYZ(r, g, b);
    return XYZToLAB(xyz);
}

// ============================================================================
// Distance in LAB space (squared Euclidean ~ ΔE²)
// ============================================================================

static f32 LABDistSq(const LABColor& a, const LABColor& b) {
    f32 dL = a.L - b.L;
    f32 da = a.a - b.a;
    f32 db = a.b - b.b;
    return dL * dL + da * da + db * db;
}

// ============================================================================
// K-means++ Initialization
// ============================================================================

static void KMeansPPInit(
    const Vector<LABColor>& samples, u32 K,
    Vector<LABColor>& centroids, std::mt19937& rng)
{
    u32 N = static_cast<u32>(samples.size());
    Vector<f32> dist(N, std::numeric_limits<f32>::max());

    // First centroid: random sample
    std::uniform_int_distribution<u32> pick(0, N - 1);
    centroids.push_back(samples[pick(rng)]);

    for (u32 k = 1; k < K; ++k) {
        // Update distances to nearest centroid
        for (u32 i = 0; i < N; ++i) {
            f32 d = LABDistSq(samples[i], centroids[k - 1]);
            if (d < dist[i]) dist[i] = d;
        }

        // Weighted random pick proportional to dist²
        f32 total = std::accumulate(dist.begin(), dist.end(), 0.0f);
        if (total <= 0.0f) {
            centroids.push_back(samples[pick(rng)]);
            continue;
        }

        std::uniform_real_distribution<f32> uni(0.0f, total);
        f32 r = uni(rng);
        f32 cumul = 0.0f;
        u32 chosen = 0;
        for (u32 i = 0; i < N; ++i) {
            cumul += dist[i];
            if (cumul >= r) { chosen = i; break; }
        }
        centroids.push_back(samples[chosen]);
    }
}

// ============================================================================
// Public: ClusterTextureColors
// ============================================================================

ClusterResult ClusterTextureColors(
    const u8* pixels, u32 width, u32 height, u32 channels,
    bool isSRGB, u32 K, u32 maxIterations)
{
    u32 totalPixels = width * height;

    // Subsample stride: target ~10000 samples
    u32 stride = std::max(1u, static_cast<u32>(std::sqrt(
        static_cast<f32>(totalPixels) / 10000.0f)));

    // Collect samples in LAB space
    Vector<LABColor> samples;
    samples.reserve(totalPixels / (stride * stride) + 1);

    for (u32 y = 0; y < height; y += stride) {
        for (u32 x = 0; x < width; x += stride) {
            u32 idx = (y * width + x) * channels;

            // Skip transparent pixels
            if (channels >= 4 && pixels[idx + 3] < 128) continue;

            f32 r = static_cast<f32>(pixels[idx + 0]) / 255.0f;
            f32 g = static_cast<f32>(pixels[idx + 1]) / 255.0f;
            f32 b = (channels >= 3) ? static_cast<f32>(pixels[idx + 2]) / 255.0f : r;

            if (isSRGB) {
                r = SRGBToLinear(r);
                g = SRGBToLinear(g);
                b = SRGBToLinear(b);
            }

            samples.push_back(RGBToLAB(r, g, b));
        }
    }

    // Edge case: no valid samples
    if (samples.empty()) {
        return { { {50.0f, 0.0f, 0.0f} }, {1}, 0 };
    }

    // Clamp K to sample count
    K = std::min(K, static_cast<u32>(samples.size()));
    if (K == 0) K = 1;

    // K-means++ init
    std::mt19937 rng(42);
    Vector<LABColor> centroids;
    centroids.reserve(K);
    KMeansPPInit(samples, K, centroids, rng);

    // Assignment buffer
    Vector<u32> assignment(samples.size(), 0);

    // K-means iterations
    for (u32 iter = 0; iter < maxIterations; ++iter) {
        // Assign each sample to nearest centroid
        for (u32 i = 0; i < samples.size(); ++i) {
            f32 bestDist = std::numeric_limits<f32>::max();
            for (u32 k = 0; k < K; ++k) {
                f32 d = LABDistSq(samples[i], centroids[k]);
                if (d < bestDist) { bestDist = d; assignment[i] = k; }
            }
        }

        // Recompute centroids
        Vector<LABColor> newCentroids(K, {0, 0, 0});
        Vector<u32> counts(K, 0);

        for (u32 i = 0; i < samples.size(); ++i) {
            u32 k = assignment[i];
            newCentroids[k].L += samples[i].L;
            newCentroids[k].a += samples[i].a;
            newCentroids[k].b += samples[i].b;
            counts[k]++;
        }

        f32 maxShift = 0.0f;
        for (u32 k = 0; k < K; ++k) {
            if (counts[k] > 0) {
                newCentroids[k].L /= static_cast<f32>(counts[k]);
                newCentroids[k].a /= static_cast<f32>(counts[k]);
                newCentroids[k].b /= static_cast<f32>(counts[k]);
            } else {
                newCentroids[k] = centroids[k]; // keep empty cluster
            }
            f32 shift = LABDistSq(centroids[k], newCentroids[k]);
            if (shift > maxShift) maxShift = shift;
        }

        centroids = std::move(newCentroids);

        // Converged when max centroid shift < 0.5 ΔE (i.e., 0.25 ΔE²)
        if (maxShift < 0.25f) break;
    }

    // Build result
    Vector<u32> sizes(K, 0);
    for (u32 k : assignment) sizes[k]++;

    u32 dominant = 0;
    for (u32 k = 1; k < K; ++k) {
        if (sizes[k] > sizes[dominant]) dominant = k;
    }

    return { std::move(centroids), std::move(sizes), dominant };
}

} // namespace spectraforge
