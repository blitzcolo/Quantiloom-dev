/**
 * @file RgbToSpectrum.hpp
 * @brief Jakob & Hanika spectral upsampling: an RGB reflectance as a smooth spectrum
 *
 * Internal, like CIE_CMF_Data.hpp and D65Illuminant.hpp beside it, and for the
 * same reason: the table is an implementation detail of the colour conversions.
 *
 * WHAT THIS REPLACES. The shaders used to spread an RGB reflectance across the
 * band as a normalised sum of three Gaussians on the sRGB primaries. Measured
 * against this repo's own observer and D65, that mapping is off by a mean of
 * 21 CIELab units over the sRGB cube and 35 over saturated colours -- ten to
 * fifteen just-noticeable differences. It is not a small error hiding in the
 * tails; it is the wrong colour.
 *
 * THE MODEL. A reflectance is a sigmoid of a quadratic in wavelength:
 *
 *     R(lambda) = s( c0 t^2 + c1 t + c2 ),   t = (lambda - 380) / 400
 *     s(x)      = 1/2 + x / (2 sqrt(1 + x^2))
 *
 * Three numbers per colour. Two properties follow from the shape alone and are
 * why this form was chosen over a free spectrum:
 *
 *   - s maps R -> (0,1), so the reflectance is bounded without a clamp. The
 *     Gaussian version needed `clamp(..., 0.0, 1.5)` and could return more than
 *     one for an HDR base colour, which is a surface returning more light than
 *     reached it.
 *   - A quadratic through a sigmoid is smooth and at most bimodal, which is
 *     what real reflectances look like. Fitting an unconstrained spectrum to
 *     three numbers gives spiky metamers that ring under a non-D65 illuminant.
 *
 * ACCURACY. Fitting is exact: for any colour inside the sRGB gamut the solver
 * drives the CIELab residual to zero, so the only error left is the table's own
 * interpolation. Measured on 200k random colours plus the gamut corners:
 *
 *     res   coeff memory   mean dE     p99    worst
 *      32       1.1 MB      0.098    0.453    3.49
 *      48       3.8 MB      0.043    0.201    2.05
 *      64       9.0 MB      0.024    0.112    1.23
 *
 * A just-noticeable difference is about 2.3. 64 is what the reference
 * implementation ships and is what kRgbToSpectrumResolution is set to; 48 is
 * the smallest table whose *worst* colour is still under one JND, if the memory
 * ever matters.
 *
 * Reproduced by `colour_lab --lut-sweep`, which drives MeasureAccuracy below:
 * 200k colours drawn uniformly from the unit cube plus the eight corners, seed
 * in the tool. An earlier edition of this table quoted a p99 around a third
 * higher at every resolution while agreeing on mean and worst -- the tail is
 * the part a sampling distribution moves, and that edition did not record which
 * one it used. These numbers name theirs.
 *
 * OUTSIDE THE FIT DOMAIN THE MODEL IS DANGEROUS, not merely useless. The
 * quadratic keeps growing, so past 780 nm the sigmoid saturates -- to 1 when c0
 * is positive, which for a saturated warm colour it is. SheenChair's mango
 * velvet comes back as reflectance 0.9998 at 1.2 um and 1.0000 at 10 um. In a
 * thermal render that is a mirror where a wall should be: emissivity zero, no
 * self-emission. The old Gaussian decayed to zero instead, which is wrong in
 * the opposite direction and much less alarming.
 *
 * So EvaluateRgbSpectrum clamps lambda to [380, 780] and callers outside the
 * visible band must not reach it at all. That is a rule the shaders already had
 * for sheen and diffuse transmission (`allowRgbUpsample`); base colour now
 * follows it too.
 *
 * @see Jakob & Hanika, "A Low-Dimensional Function Space for Efficient Spectral
 *      Upsampling", Computer Graphics Forum 38(2), 2019.
 */

#pragma once

#include "core/Types.hpp"

#include <glm/glm.hpp>

namespace quantiloom {

/// The band the polynomial was fitted over -- the observer's own support, which
/// is also D65Illuminant.hpp's. Nothing outside it was ever a target.
inline constexpr f32 RGB2SPEC_LAMBDA_MIN = 380.0f;
inline constexpr f32 RGB2SPEC_LAMBDA_MAX = 780.0f;
inline constexpr f32 RGB2SPEC_LAMBDA_RANGE = RGB2SPEC_LAMBDA_MAX - RGB2SPEC_LAMBDA_MIN;

/// Table resolution per axis. See the accuracy table above before changing it;
/// the GPU buffer and the disk cache are both sized from this and the cache
/// header records it, so a change invalidates caches rather than corrupting
/// them.
inline constexpr u32 kRgbToSpectrumResolution = 64;

/**
 * @brief The three polynomial coefficients, in the normalised-t domain
 *
 * Deliberately not remapped to nanometres the way the reference implementation
 * stores them. In nm the leading coefficient is around 1e-5 and the constant
 * around 1e+1, six orders apart, and the shader would evaluate
 * `c0*lambda*lambda` with lambda near 700 -- a product of a tiny number and a
 * large one, in fp32, inside a path tracer. Keeping t in [0,1] puts all three
 * coefficients within a couple of orders of each other and costs one multiply.
 */
struct RgbSpectrumCoeffs {
    f32 c0 = 0.0f;
    f32 c1 = 0.0f;
    f32 c2 = 0.0f;
};

/**
 * @brief Reflectance at one wavelength
 *
 * lambda is clamped to the fit domain, so a caller that reaches here from an
 * infrared band gets the 780 nm value rather than the runaway described in the
 * file comment. That clamp is a guard rail, not a licence: an infrared caller
 * is asking a question this model cannot answer either way.
 */
[[nodiscard]] f32 EvaluateRgbSpectrum(const RgbSpectrumCoeffs& c, f32 lambda_nm);

/**
 * @brief The exact coefficients for an achromatic colour
 *
 * s(x) = g has the closed form x = (g - 1/2) / sqrt(g (1 - g)), so a grey needs
 * no table at all: c0 = c1 = 0 and the spectrum is flat at exactly g.
 *
 * This is not an optimisation. Every dielectric without KHR_materials_specular
 * has F0 = (0.04, 0.04, 0.04), and the scenes both render gates use are grey --
 * shadow_scene_open is base colour 0.5. Returning g through a closed form
 * rather than through eight table fetches and a trilinear blend is what lets
 * those renders stay bit-identical across this change.
 */
[[nodiscard]] RgbSpectrumCoeffs AchromaticRgbSpectrum(f32 g);

/**
 * @class RgbToSpectrumTable
 * @brief The fitted coefficient table, built once and cached
 *
 * Built rather than shipped, following BRDFLutGenerator: the fit depends on
 * CIE_CMF_Data.hpp and D65Illuminant.hpp, and a checked-in binary would drift
 * from them silently the first time either is corrected. Building takes a few
 * seconds and the versioned cache means it happens once per machine.
 *
 * LAYOUT. Three sub-tables, selected by which RGB channel is largest, because
 * the remaining two channels divided by that maximum are what actually
 * determine the spectrum's shape. Within a sub-table the axes are
 *
 *     x = rgb[(maxc + 1) % 3] / rgb[maxc]     uniform in [0,1]
 *     y = rgb[(maxc + 2) % 3] / rgb[maxc]     uniform in [0,1]
 *     z = rgb[maxc]                           warped, see ZNode
 *
 * and the flat index is ((maxc * res + zi) * res + yi) * res + xi.
 *
 * The z axis is warped by smoothstep applied twice because the coefficients
 * move fastest as a colour approaches black or white -- c2 runs off toward
 * -inf and +inf at the two ends -- and a uniform axis spends its resolution in
 * the middle where nothing happens.
 */
class RgbToSpectrumTable {
public:
    /**
     * @brief Fit the whole table
     *
     * Multi-threaded over (maxc, y) slices, which write disjoint regions and
     * need no synchronisation. Each column in z is solved by continuation --
     * the solution at one brightness warm-starts the next -- outward from
     * res/5 in both directions, because a mid-brightness colour is the easiest
     * place to start and the two ends are where the fit is hardest.
     */
    [[nodiscard]] static Result<RgbToSpectrumTable, String> Build(u32 resolution);

    /// Read a cache, verifying magic, version and resolution; build and write
    /// one if that fails for any reason. A corrupt or stale cache costs a
    /// rebuild, never a wrong table.
    [[nodiscard]] static Result<RgbToSpectrumTable, String> LoadOrBuild(
        const String& cachePath, u32 resolution);

    [[nodiscard]] bool Save(const String& path) const;

    /**
     * @brief Trilinear fetch, matching the shader's arithmetic exactly
     *
     * Achromatic input takes AchromaticRgbSpectrum and never touches the table.
     * Components are expected in [0,1]; an HDR colour is the caller's business
     * to scale down first (see the note on RGBUnboundedSpectrum in the shader),
     * and out-of-range input is clamped rather than extrapolated.
     */
    [[nodiscard]] RgbSpectrumCoeffs Lookup(const glm::vec3& rgb) const;

    [[nodiscard]] u32 Resolution() const { return m_res; }

    /// 3 * res^3 * 3 floats, coefficient-major. The GPU upload pads each triple
    /// to a float4; that padding is not stored here.
    [[nodiscard]] const Vector<f32>& Data() const { return m_data; }

    /// The warped z coordinate of node i: smoothstep(smoothstep(i / (res-1))).
    /// The shader inverts this in closed form rather than searching, so this
    /// and that inverse are a matched pair.
    [[nodiscard]] static f32 ZNode(u32 i, u32 res);

private:
    u32 m_res = 0;
    Vector<f32> m_data;
};

/**
 * @brief What a table's interpolation costs, in CIELab
 *
 * The fit at a lattice node is exact -- the solver drives the Lab residual to
 * zero for any in-gamut colour -- so everything this reports is the trilinear
 * blend between nodes, which is the only thing the resolution controls.
 */
struct RgbToSpectrumAccuracy {
    u32 resolution = 0;
    u64 coefficientBytes = 0;
    u32 samples = 0;
    f64 meanDeltaE = 0.0;
    f64 p99DeltaE = 0.0;
    f64 worstDeltaE = 0.0;
    glm::vec3 worstColour{0.0f};
};

/**
 * @brief Measure a table against the colours it claims to reproduce
 *
 * Lives here rather than in a tool because the comparison has to use the
 * fitter's own observer, illuminant and Lab conversion. Those are file-static
 * in RgbToSpectrum.cpp, and a measurement that reimplemented them elsewhere
 * would drift from the thing it measures the first time either is corrected --
 * which is the same argument that keeps the table built rather than shipped.
 *
 * Sampling is `randomSamples` colours drawn uniformly from the unit cube with
 * the given seed, plus the eight cube corners, which are the hardest colours
 * in the gamut and would be missed by any finite random draw.
 */
[[nodiscard]] RgbToSpectrumAccuracy MeasureAccuracy(const RgbToSpectrumTable& table,
                                                    u32 randomSamples, u64 seed);

}  // namespace quantiloom
