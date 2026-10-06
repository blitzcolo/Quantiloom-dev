#ifndef QL_CAMERA_PROJECTION_HLSLI
#define QL_CAMERA_PROJECTION_HLSLI

// Layout matches RayTracingPipeline::SetCameraProjection, float4 x 4.
[[vk::binding(32, 0)]] StructuredBuffer<float4> projectionConfig;

float2 BrownProjection(float2 v, float4 k, float k3, out float3 j) {
    float r2 = dot(v, v);
    float a = 1 + k.x * r2 + k.y * r2 * r2 + k3 * r2 * r2 * r2;
    float da = 2 * (k.x + 2 * k.y * r2 + 3 * k3 * r2 * r2);
    j = float3(a + v.x * v.x * da + 2 * k.z * v.y + 6 * k.w * v.x,
               v.x * v.y * da + 2 * k.z * v.x + 2 * k.w * v.y,
               a + v.y * v.y * da + 6 * k.z * v.y + 2 * k.w * v.x);
    return float2(v.x * a + 2 * k.z * v.x * v.y + k.w * (r2 + 2 * v.x * v.x),
                  v.y * a + k.z * (r2 + 2 * v.y * v.y) + 2 * k.w * v.x * v.y);
}

float FishProjection(float t, float4 k) {
    float t2 = t * t;
    return t * (1 + t2 * (k.x + t2 * (k.y + t2 * (k.z + t2 * k.w))));
}

bool NativePixelDirection(float2 pixel, out float3 localDirection) {
    float4 header = projectionConfig[0];
    float4 intrinsics = projectionConfig[1];
    float4 k = projectionConfig[2];
    float k3 = projectionConfig[3].x;
    float2 target = (pixel - intrinsics.zw) / intrinsics.xy;
    float2 v = target;
    uint model = (uint)header.y;
    if (model == 1) {
        bool converged = false;
        [loop] for (uint i = 0; i < 40; ++i) {
            float3 j;
            float2 error = BrownProjection(v, k, k3, j) - target;
            if (length(error * intrinsics.xy) < 0.001) {
                converged = true;
                break;
            }
            float determinant = j.x * j.z - j.y * j.y;
            if (determinant <= 1e-10 || !isfinite(determinant))
                break;
            float2 step = float2(j.z * error.x - j.y * error.y,
                                 j.x * error.y - j.y * error.x) / determinant;
            float scale = 1;
            [loop] for (uint backtrack = 0; backtrack < 10; ++backtrack) {
                float3 unused;
                if (length(BrownProjection(v - step * scale, k, k3, unused) - target) <
                    length(error))
                    break;
                scale *= 0.5;
            }
            v -= step * scale;
        }
        if (!converged) {
            localDirection = 0;
            return false;
        }
    } else if (model == 2) {
        float r = length(target);
        if (r > FishProjection(header.z, k)) {
            localDirection = 0;
            return false;
        }
        float lo = 0, hi = header.z;
        [loop] for (uint i = 0; i < 32; ++i) {
            float t = (lo + hi) * 0.5;
            if (FishProjection(t, k) < r)
                lo = t;
            else
                hi = t;
        }
        if (r > 1e-8)
            v *= tan((lo + hi) * 0.5) / r;
    }
    localDirection = normalize(float3(v, 1));
    return all(isfinite(localDirection));
}
#endif
