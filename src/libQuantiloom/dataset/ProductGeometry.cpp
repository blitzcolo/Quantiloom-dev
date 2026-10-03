#include "dataset/ProductGeometry.hpp"
#include <nlohmann/json.hpp>
#include <cmath>

namespace quantiloom::dataset {
Result<String, String> ProductGeometry::ToJson() const {
    using Json = nlohmann::json;
    if (version != 1 || width == 0 || height == 0 ||
        !std::isfinite(referenceTimeSeconds) || !std::isfinite(worldUnitsToMeters) ||
        worldUnitsToMeters <= 0.0 || camera.projection > 1)
        return Result<String, String>::Err("invalid product geometry");
    const glm::dvec3 r(camera.right), d(-camera.up), f(camera.forward), o(camera.origin);
    const auto finite = [](const glm::dvec3& v) {
        return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
    };
    if (!finite(r) || !finite(d) || !finite(f) || !finite(o) ||
        std::abs(glm::dot(r, r) - 1.0) > 1e-5 ||
        std::abs(glm::dot(d, d) - 1.0) > 1e-5 ||
        std::abs(glm::dot(f, f) - 1.0) > 1e-5 ||
        std::abs(glm::dot(r, d)) > 1e-5 || std::abs(glm::dot(r, f)) > 1e-5 ||
        std::abs(glm::dot(d, f)) > 1e-5 || !std::isfinite(camera.aspectRatio) ||
        camera.aspectRatio <= 0.0f)
        return Result<String, String>::Err("invalid camera basis");
    Json out = {{"version", nativeProjection ? 2u : version}, {"width", width}, {"height", height},
        {"reference_time_s", referenceTimeSeconds}, {"kind", "instantaneous_geometry"},
        {"world_units_to_meters", worldUnitsToMeters},
        {"camera_axes", "right_down_forward"}, {"pixel_origin", "top_left"},
        {"pixel_center_offset", {0.5, 0.5}}, {"matrix_order", "row_major"},
        {"world_to_camera", {r.x,r.y,r.z,-glm::dot(r,o),
                             d.x,d.y,d.z,-glm::dot(d,o),
                             f.x,f.y,f.z,-glm::dot(f,o),0.0,0.0,0.0,1.0}},
        {"camera_to_world", {r.x,d.x,f.x,o.x,r.y,d.y,f.y,o.y,r.z,d.z,f.z,o.z,0.0,0.0,0.0,1.0}}
    };
    if (camera.projection == 0) {
        if (!std::isfinite(camera.fovScale) || camera.fovScale <= 0.0f)
            return Result<String, String>::Err("invalid perspective projection");
        const f64 fy = height / (2.0 * camera.fovScale);
        const f64 fx = width / (2.0 * camera.fovScale * camera.aspectRatio);
        out["projection"] = "perspective";
        out["intrinsics"] = {fx,0.0,width*0.5,0.0,fy,height*0.5,0.0,0.0,1.0};
        out["fov_y_radians"] = 2.0 * std::atan(camera.fovScale);
        out["fov_x_radians"] = 2.0 * std::atan(camera.fovScale * camera.aspectRatio);
        if (nativeProjection) {
            const auto valid = camera::ValidateProjection(*nativeProjection, width, height);
            if (!valid) return Result<String,String>::Err(valid.error());
            const auto& p=*nativeProjection;
            out["intrinsics"]={p.fx,0.0,p.cx,0.0,p.fy,p.cy,0.0,0.0,1.0};
            out["camera_model"]=camera::ProjectionModelName(p.model);
            out["distortion"]={{"model",camera::ProjectionModelName(p.model)},
                {"coefficients",p.coefficients},{"max_theta_radians",p.maxThetaRadians},
                {"coefficient_order",p.model==camera::ProjectionModel::Fisheye ?
                    "k1_k2_k3_k4" : "k1_k2_p1_p2_k3"}};
            out["fov_y_radians"]=nullptr;
            out["fov_x_radians"]=nullptr;
            out["valid_domain"]="native_grid_and_invertible_lens_field";
        }
    } else {
        if (!std::isfinite(camera.orthoHeight) || camera.orthoHeight <= 0.0f)
            return Result<String, String>::Err("invalid orthographic projection");
        out["projection"] = "orthographic";
        out["film_height_world_units"] = camera.orthoHeight;
        out["film_width_world_units"] = static_cast<f64>(camera.orthoHeight) * camera.aspectRatio;
        out["intrinsics"] = nullptr;
    }
    return out.dump();
}
} // namespace quantiloom::dataset
