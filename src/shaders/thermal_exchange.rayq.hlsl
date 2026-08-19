// ============================================================================
// Quantiloom - Thermal Exchange Precompute (inline ray query)
// ============================================================================
// Who sees whom. One thread per surface element, casting a cosine-weighted
// hemisphere of rays from the element's centre and recording what each one
// finds: another element, or the sky.
//
// That distribution is the point. Radiative exchange between two surfaces
// goes as cos(theta_i) cos(theta_j) / (pi r^2) integrated over both, and
// sampling directions proportional to cos(theta_i) makes the estimator of
// element i's view factors a plain histogram of hits -- no weights, no
// division by a pdf. What the geometry does the sampling already did.
//
// The result is a fraction of the hemisphere per element, which is exactly
// what the energy balance multiplies T^4 by. The host reduces the hit records
// into sparse rows.
//
// A second pass answers the other geometric question -- whether the sun
// reaches this element -- because it is the same ray cast in a fixed
// direction, and doing it here means the solver never needs the acceleration
// structure.
//
// Compiled as cs_6_5 (RayQuery needs SM 6.5); see the .rayq.hlsl rule in
// CMakeLists.txt. Deliberately does NOT include common.hlsli: that header
// declares the full RT pipeline's resources, which this small set does not
// bind.
// ============================================================================

[[vk::binding(0, 0)]] RaytracingAccelerationStructure sceneTlas;

// One per element, in the order BuildThermalMesh produced them.
struct ThermalElementGpu {
    // `centre`, not `centroid`: the latter is an HLSL interpolation modifier
    // and a member by that name does not parse.
    float3 centre;
    float  area;
    float3 normal;
    uint   materialId;
};
[[vk::binding(1, 0)]] StructuredBuffer<ThermalElementGpu> elements;

// instanceElementBase[instanceIndex] + primitiveIndex is the element a hit
// landed on. The same mapping the closest-hit shader uses to find a
// temperature, so a hit here and a shade there agree about which triangle
// they are talking about.
[[vk::binding(2, 0)]] StructuredBuffer<uint> instanceElementBase;

// Hit records: rayCount entries per element, 0xFFFFFFFF where the ray escaped.
// A histogram rather than a matrix, because the host knows how to make one
// sparse and a shader writing scattered increments would need atomics into
// something n^2.
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> hitRecords;

// One per element: the fraction of the sun disc samples that reached it.
[[vk::binding(4, 0)]] RWStructuredBuffer<float> sunVisibility;

// Per material, the fraction of its area that is actually there: 1 for an
// opaque surface, and for alphaMode MASK or BLEND the mean of its base colour
// alpha, computed on the CPU where the texels still live.
//
// A mean rather than a per-texel test, which is a deliberate choice and not a
// shortcut. A view factor is an area integral -- what fraction of element i's
// hemisphere element j subtends -- estimated here by a histogram of a few
// hundred cosine-weighted rays. Resolving each ray against its own texel would
// need this pass to carry the whole bindless texture set of the render
// pipeline, and would answer a question finer than the estimator can hear: over
// many rays the hit points are spread across the occluder, so their expectation
// is exactly this mean. What it cannot represent is a leaf whose holes are all
// on one side, which no view factor at this resolution distinguishes anyway.
[[vk::binding(5, 0)]] StructuredBuffer<float> materialCoverage;

struct ExchangePushConstants {
    float3 sunDirection;   // from surface toward the sun, normalised
    uint   elementCount;
    uint   rayCount;       // hemisphere rays per element; 0 = sun-only pass
    uint   sunRayCount;    // rays toward the sun, for a soft shadow edge
    float  sunAngularRadius;  // radians; 0.00465 is the real sun
    float  rayOffset;      // how far off the surface a ray starts
    uint   sunOutputOffset;   // element offset into sunVisibility for this direction
};
[[vk::push_constant]] ExchangePushConstants pc;

// ----------------------------------------------------------------------------
// Sampling
// ----------------------------------------------------------------------------

// Hammersley: a low-discrepancy pair, which for a fixed ray count gives a
// noticeably smoother set of view factors than white noise at no cost. The
// sequence is the same for every element, and that is fine -- the tangent
// frame it is rotated into is not.
float RadicalInverseVdC(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10;
}

float2 Hammersley(uint i, uint n) {
    return float2(float(i) / float(n), RadicalInverseVdC(i));
}

// An orthonormal basis around n, branchless (Duff et al. 2017).
void BuildBasis(float3 n, out float3 t, out float3 b) {
    const float sign = n.z >= 0.0 ? 1.0 : -1.0;
    const float a = -1.0 / (sign + n.z);
    const float c = n.x * n.y * a;
    t = float3(1.0 + sign * n.x * n.x * a, sign * c, -sign * n.x);
    b = float3(c, sign + n.y * n.y * a, -n.y);
}

// Cosine-weighted, by Malley's method: a uniform point on the disc lifted to
// the hemisphere. This is what makes the histogram a view factor.
float3 SampleCosineHemisphere(float2 u, float3 n) {
    const float r = sqrt(u.x);
    const float phi = 6.2831853071795864 * u.y;
    const float z = sqrt(max(0.0, 1.0 - u.x));

    float3 t, b;
    BuildBasis(n, t, b);
    return normalize(t * (r * cos(phi)) + b * (r * sin(phi)) + n * z);
}

// A hash with no state to carry, for the coverage coin below. The candidate
// order along a ray is not promised, so the seed is built from things that do
// not depend on it: which element is casting, which of its rays this is, and
// which triangle is being considered.
uint CoverageHash(uint element, uint rayIndex, uint instance, uint primitive) {
    uint s = element * 73856093u ^ rayIndex * 19349663u ^
             instance * 2654435761u ^ primitive * 40503u;
    s = s * 747796405u + 2891336453u;
    s = ((s >> ((s >> 28) + 4u)) ^ s) * 277803737u;
    return (s >> 22) ^ s;
}

uint TraceForElement(float3 origin, float3 direction, uint element, uint rayIndex) {
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = 0.0;
    ray.TMax = 1e6;

    // Not FORCE_OPAQUE any more: geometry whose material is alphaMode MASK or
    // BLEND is built non-opaque, and a leaf with holes in it should let
    // radiation through them exactly as it lets light through.
    RayQuery<RAY_FLAG_NONE> query;
    query.TraceRayInline(sceneTlas, RAY_FLAG_NONE, 0xFF, ray);
    while (query.Proceed()) {
        if (query.CandidateType() != CANDIDATE_NON_OPAQUE_TRIANGLE) {
            continue;
        }

        // The occluder's material, by the same instance-plus-primitive mapping
        // a committed hit uses below.
        float coverage = 1.0;
        const uint candidateBase = instanceElementBase[query.CandidateInstanceIndex()];
        if (candidateBase != 0xFFFFFFFFu) {
            const uint candidateElement = candidateBase + query.CandidatePrimitiveIndex();
            if (candidateElement < pc.elementCount) {
                coverage = materialCoverage[elements[candidateElement].materialId];
            }
        }

        if (coverage >= 1.0) {
            query.CommitNonOpaqueTriangleHit();
            continue;
        }
        const float xi = float(CoverageHash(element, rayIndex,
                                            query.CandidateInstanceIndex(),
                                            query.CandidatePrimitiveIndex())) *
                         (1.0 / 4294967296.0);
        if (xi < coverage) {
            query.CommitNonOpaqueTriangleHit();
        }
    }

    if (query.CommittedStatus() != COMMITTED_TRIANGLE_HIT) {
        return 0xFFFFFFFFu;  // the sky
    }
    const uint instance = query.CommittedInstanceIndex();
    const uint base = instanceElementBase[instance];
    if (base == 0xFFFFFFFFu) {
        return 0xFFFFFFFFu;  // an instance with no elements: treat as sky
    }
    return base + query.CommittedPrimitiveIndex();
}

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    const uint e = tid.x;
    if (e >= pc.elementCount) return;

    const ThermalElementGpu element = elements[e];

    // A degenerate triangle has no hemisphere to sample. Its rows stay empty
    // and its sun visibility zero, which the solver skips anyway.
    if (element.area <= 0.0) {
        if (pc.rayCount > 0) {
            for (uint r = 0; r < pc.rayCount; ++r) {
                hitRecords[e * pc.rayCount + r] = 0xFFFFFFFFu;
            }
        }
        sunVisibility[pc.sunOutputOffset + e] = 0.0;
        return;
    }

    // Offset along the normal, so a ray does not immediately re-hit the
    // triangle it left. Scaled with the element rather than fixed: a scene in
    // millimetres and a scene in metres need different epsilons, and the area
    // is the only length this shader has.
    const float3 origin = element.centre + element.normal * pc.rayOffset;

    // Hemisphere pass: only when rayCount > 0. A sun-only batch pass sets
    // rayCount to zero and skips the hemisphere entirely.
    if (pc.rayCount > 0) {
        for (uint r = 0; r < pc.rayCount; ++r) {
            const float2 u = Hammersley(r, pc.rayCount);
            const float3 direction = SampleCosineHemisphere(u, element.normal);
            hitRecords[e * pc.rayCount + r] = TraceForElement(origin, direction, e, r);
        }
    }

    // The sun, if there is one above this element's horizon. Several rays
    // across the disc rather than one, so an element at the edge of a shadow
    // gets a fraction instead of a step -- which is what makes a shadow
    // boundary in the temperature field as soft as the geometry says.
    float visible = 0.0;
    const float cosSun = dot(element.normal, pc.sunDirection);
    if (cosSun > 0.0 && pc.sunRayCount > 0) {
        float3 t, b;
        BuildBasis(pc.sunDirection, t, b);
        for (uint s = 0; s < pc.sunRayCount; ++s) {
            const float2 u = Hammersley(s, pc.sunRayCount);
            const float radius = pc.sunAngularRadius * sqrt(u.x);
            const float phi = 6.2831853071795864 * u.y;
            const float3 direction = normalize(pc.sunDirection +
                                               t * (radius * cos(phi)) +
                                               b * (radius * sin(phi)));
            // Offset the ray index past the hemisphere's, so a sun ray and a
            // hemisphere ray never draw the same coin for the same occluder.
            if (TraceForElement(origin, direction, e, pc.rayCount + s) == 0xFFFFFFFFu) {
                visible += 1.0;
            }
        }
        visible /= float(pc.sunRayCount);
    }
    sunVisibility[pc.sunOutputOffset + e] = visible;
}
