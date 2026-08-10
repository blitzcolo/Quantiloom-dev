// ============================================================================
// Quantiloom - Sample Generation
// ============================================================================
// Padded Owen-scrambled Sobol' sampling, plus the PCG stream it falls back to.
//
// WHY THIS EXISTS
//
// Every decision a path makes -- where in the pixel to start, which wavelength
// to carry, which emitter to sample, which direction to bounce -- used to draw
// from PCG, which is white noise. White noise clumps: N independent points
// leave gaps and pile up, and the estimator's variance falls as 1/N with a
// constant set by that clumping. A stratified point set covers the domain
// evenly instead, which lowers the constant without touching the expectation.
//
// UNBIASEDNESS
//
// This changes only the joint distribution of the samples, never the marginal.
// Owen scrambling permutes the base-2 digit tree of a coordinate: each point of
// a scrambled sequence is still exactly uniform on [0,1), so E[f(x_i)] is
// unchanged for every i and the estimator's mean is what it always was. What
// changes is that the points are negatively correlated with each other, which
// is where the variance reduction comes from. No approximation, no tuning
// parameter, no bias to trade away.
//
// WHY PADDED, AND NOT NINE DIMENSIONS OF SOBOL'
//
// The obvious construction is one Sobol' dimension per decision -- dims 0..8
// for jitter.x, jitter.y, lambda, emitter, u, v, lobe, dir.x, dir.y. That is
// worse than it looks. Sobol' is a (t,s)-sequence whose t degrades with
// dimension, and only the low projections are properly stratified: checking the
// elementary-interval property directly (scripts/gen_sobol.py) shows the pair
// (0,1) passing and the pair (4,5) -- which would have been the triangle uv --
// failing at 2^-4 x 2^-4. Those samples would have been uniform on each axis
// separately and clumped jointly, which is most of the benefit gone.
//
// So this uses padding instead (PBRT-v4's PaddedSobolSampler; Burley 2020).
// One 2D Sobol' point set -- dimensions 0 and 1, the pair that genuinely is a
// (0,2)-sequence -- is reused for every decision, with an INDEPENDENT Owen
// scramble seed per decision and per pixel. Each 2D decision then gets a fully
// stratified 2D point set, decisions are decorrelated from each other by their
// scrambles, and neighbouring pixels are decorrelated the same way.
//
// WHY EACH SLOT ALSO SHUFFLES THE SAMPLE INDEX
//
// Re-scrambling the same base sequence is NOT enough to make two slots
// independent, and getting this wrong is silent and expensive.
//
// Owen scrambling permutes the digit tree: output digit k is the input digit k
// flipped by a function of the digits ABOVE it. In particular the leading digit
// is flipped by a constant that depends on the seed alone. So for two slots A
// and B scrambling the same point x_i, the leading bits are x_i's leading bit
// XOR a and XOR b -- which means they are equal for every i, or opposite for
// every i. The slots are perfectly locked, whatever the seeds. Measured on the
// real construction (scripts/gen_sobol.py): 240 of 256 cells of the joint 2D
// histogram empty, leading-bit agreement exactly 100% or exactly 0%.
//
// What that does to a renderer is worse than it sounds, because a threshold
// test on one slot then partitions the samples by their leading bit -- and on
// that partition, every other slot's leading bit is CONSTANT. Choosing a lobe
// with `u < qSpec` and then drawing a direction left the specular branch
// sampling half of the hemisphere and never the other half. It cost 3.6x on a
// scene with mixed lobes, and it looked like plain noise.
//
// Permuting the sample index per slot fixes it, because the permutation acts
// on WHICH point of the sequence a slot sees rather than on the point's digits.
// The permutation is itself an Owen scramble, of the index, which is what keeps
// progressivity: every 2^k prefix still lands one sample per stratum in 1D and
// one per cell in 2D, verified for k = 2..10. So an interactive render stopped
// at an arbitrary sample count is still stratified at the count it reached.
//
// The whole construction is verified in scripts/gen_sobol.py: matrix structure,
// 1D balance to 2^14, the (0,2) property of the pair, that the scramble is a
// bijection, that scrambling PRESERVES the (0,2) property rather than merely
// producing plausible-looking uniform numbers, that two slots are jointly
// uniform, and that prefixes stay stratified.
//
// References:
//   Owen, "Randomly Permuted (t,m,s)-Nets and (t,s)-Sequences", 1995
//   Burley, "Practical Hash-based Owen Scrambling", JCGT 9(4), 2020
//   Joe & Kuo, "Constructing Sobol sequences with better two-dimensional
//     projections", SIAM J. Sci. Comput. 30(5), 2008
// ============================================================================

#ifndef QUANTILOOM_SAMPLING_HLSLI
#define QUANTILOOM_SAMPLING_HLSLI

// ============================================================================
// Decision slots
// ============================================================================
// One per decision on the first bounce. Each slot is an independent padded
// copy of the sequence, so adding a slot cannot disturb the others -- unlike a
// running dimension counter, where a branch that skips a draw shifts every
// decision after it onto the wrong dimension.
//
// Deeper bounces do not use slots at all; see PathSample1D in closesthit.
// ============================================================================

#define SAMPLE_SLOT_JITTER      0u  // subpixel position          (2D, raygen)
#define SAMPLE_SLOT_LAMBDA      1u  // bounce wavelength          (1D)
#define SAMPLE_SLOT_LIGHT_PICK  2u  // which emitter              (1D)
#define SAMPLE_SLOT_LIGHT_UV    3u  // point on that emitter      (2D)
#define SAMPLE_SLOT_LOBE        4u  // diffuse or specular        (1D)
#define SAMPLE_SLOT_DIRECTION   5u  // bounce direction           (2D)

// ============================================================================
// Sobol' generator matrix, dimension 1
// ============================================================================
// Dimension 0 needs no table: its columns are 2^(31-i), which makes the XOR of
// the selected columns exactly the bit reversal of the index.
//
// Generated and verified by scripts/gen_sobol.py. The self-similar pattern is
// Pascal's triangle mod 2 and is not a coincidence -- dimension 1's recurrence
// is m_i = m_{i-1} XOR (m_{i-1} << 1).
// ============================================================================

static const uint SOBOL_DIM1[32] = {
    0x80000000u, 0xc0000000u, 0xa0000000u, 0xf0000000u,
    0x88000000u, 0xcc000000u, 0xaa000000u, 0xff000000u,
    0x80800000u, 0xc0c00000u, 0xa0a00000u, 0xf0f00000u,
    0x88880000u, 0xcccc0000u, 0xaaaa0000u, 0xffff0000u,
    0x80008000u, 0xc000c000u, 0xa000a000u, 0xf000f000u,
    0x88008800u, 0xcc00cc00u, 0xaa00aa00u, 0xff00ff00u,
    0x80808080u, 0xc0c0c0c0u, 0xa0a0a0a0u, 0xf0f0f0f0u,
    0x88888888u, 0xccccccccu, 0xaaaaaaaau, 0xffffffffu,
};

// [unroll] so every array index is a compile-time constant and the table folds
// into 32 immediate XORs. Left as a rolled loop it becomes a dynamically
// indexed private array, which spills to scratch memory -- the cost pattern
// this shader spent a whole pass removing elsewhere.
uint SobolDim1(uint index) {
    uint x = 0u;
    [unroll]
    for (uint b = 0u; b < 32u; ++b) {
        if (index & (1u << b)) {
            x ^= SOBOL_DIM1[b];
        }
    }
    return x;
}

// ============================================================================
// Owen scrambling
// ============================================================================

// Burley 2020, section 5. A nested (Owen) permutation applied to the bits from
// the top down: each bit is flipped based on a hash of the bits above it, which
// is what makes it a permutation of the digit tree rather than a hash of the
// value.
uint LaineKarrasPermutation(uint x, uint seed) {
    x += seed;
    x ^= x * 0x6c50b47cu;
    x ^= x * 0xb82f1e52u;
    x ^= x * 0xc7afe638u;
    x ^= x * 0x8d22f6e6u;
    return x;
}

// Owen scramble = reverse, permute from the bottom, reverse back. The two
// reversals turn "flip low bits based on high bits" into the nested scramble
// the theory asks for.
uint OwenScramble(uint x, uint seed) {
    x = reversebits(x);
    x = LaineKarrasPermutation(x, seed);
    return reversebits(x);
}

// Independent seed per (pixel, slot, round). Three rounds of the PCG output
// mixer: pixels must decorrelate from each other or the stratification turns
// into a visible tiling pattern, and slots must decorrelate from each other or
// the padding is not padding.
uint SampleSlotSeed(uint2 pixel, uint slot, uint sequenceSeed) {
    uint h = sequenceSeed + slot * 0x9e3779b9u;
    h = h * 747796405u + 2891336453u;
    h = ((h >> ((h >> 28) + 4u)) ^ h) * 277803737u;
    h = (h >> 22) ^ h;
    h ^= pixel.x * 0x85ebca6bu;
    h = h * 747796405u + 2891336453u;
    h = ((h >> ((h >> 28) + 4u)) ^ h) * 277803737u;
    h = (h >> 22) ^ h;
    h ^= pixel.y * 0xc2b2ae35u;
    h = h * 747796405u + 2891336453u;
    h = ((h >> ((h >> 28) + 4u)) ^ h) * 277803737u;
    return (h >> 22) ^ h;
}

// ============================================================================
// Per-slot index shuffle
// ============================================================================
// Which point of the sequence this slot sees for sample `index`. See the note
// at the top: without this, slots do not decorrelate no matter how they are
// scrambled.
//
// Shuffled within 24 bits, so the permutation is over 16.7M samples -- past any
// render -- while leaving 8 bits of headroom for the scramble to work in.
// ============================================================================

uint ShuffleSampleIndex(uint index, uint seed) {
    return (OwenScramble((index & 0x00ffffffu) << 8u, seed) >> 8u) & 0x00ffffffu;
}

// The three seeds a slot needs -- value scramble X, value scramble Y, index
// shuffle -- from one hash chain, so a slot costs one SampleSlotSeed.
uint NextSeed(uint s) {
    s = s * 747796405u + 2891336453u;
    s = ((s >> ((s >> 28) + 4u)) ^ s) * 277803737u;
    return (s >> 22) ^ s;
}

// ============================================================================
// Stratified draws
// ============================================================================
// 1 / 2^32 rather than 1 / (2^32 - 1): the result must stay strictly below 1.0
// so that a caller scaling it by a count cannot index one past the end.
// ============================================================================

float StratifiedSample1D(uint index, uint2 pixel, uint slot, uint sequenceSeed) {
    const uint seed = SampleSlotSeed(pixel, slot, sequenceSeed);
    const uint idx  = ShuffleSampleIndex(index, NextSeed(seed));
    return float(OwenScramble(reversebits(idx), seed)) * (1.0 / 4294967296.0);
}

float2 StratifiedSample2D(uint index, uint2 pixel, uint slot, uint sequenceSeed) {
    const uint seed = SampleSlotSeed(pixel, slot, sequenceSeed);
    // Independent scrambles per axis, or the two coordinates lock together the
    // same way two slots would and the (0,2) property the pair was chosen for
    // is gone.
    const uint seedY = NextSeed(seed);
    const uint idx   = ShuffleSampleIndex(index, NextSeed(seedY));

    return float2(OwenScramble(reversebits(idx), seed),
                  OwenScramble(SobolDim1(idx), seedY)) * (1.0 / 4294967296.0);
}

// ============================================================================
// PCG32 -- the unstratified stream
// ============================================================================
// Used past the first bounce, where a path's dimension is no longer a fixed
// slot: which decisions a deep vertex makes depends on what it hit, so there is
// no stable dimension to stratify along. It is also what the transmission and
// dispersion tails draw from.
// ============================================================================

uint pcg_hash(uint s) {
    s = s * 747796405u + 2891336453u;
    s = ((s >> ((s >> 28) + 4u)) ^ s) * 277803737u;
    return (s >> 22) ^ s;
}

float pcg_float(inout uint s) {
    s = pcg_hash(s);
    return float(s) * (1.0 / 4294967296.0);
}

#endif // QUANTILOOM_SAMPLING_HLSLI
