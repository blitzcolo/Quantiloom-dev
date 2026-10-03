#include "scene/CameraProjection.hpp"
#include <cmath>
#include <limits>

namespace quantiloom::camera {
namespace {
constexpr f64 halfPi = 1.57079632679489661923;
bool Finite(const glm::dvec2& v) { return std::isfinite(v.x) && std::isfinite(v.y); }
glm::dvec2 Brown(const CameraProjection& p, glm::dvec2 v, glm::dmat2* jacobian = nullptr) {
    const auto& k = p.coefficients;
    const f64 r2 = glm::dot(v,v), r4 = r2*r2;
    const f64 a = 1 + k[0]*r2 + k[1]*r4 + k[4]*r4*r2;
    if (jacobian) {
        const f64 da = 2*(k[0] + 2*k[1]*r2 + 3*k[4]*r4);
        const f64 xy = v.x*v.y*da + 2*k[2]*v.x + 2*k[3]*v.y;
        *jacobian = glm::dmat2(a+v.x*v.x*da+2*k[2]*v.y+6*k[3]*v.x, xy,
                             xy, a+v.y*v.y*da+6*k[2]*v.y+2*k[3]*v.x);
    }
    return {v.x*a + 2*k[2]*v.x*v.y + k[3]*(r2+2*v.x*v.x),
            v.y*a + k[2]*(r2+2*v.y*v.y) + 2*k[3]*v.x*v.y};
}
f64 Fish(const CameraProjection& p, f64 t, f64* derivative = nullptr) {
    const auto& k = p.coefficients;
    const f64 t2=t*t, t4=t2*t2, t6=t4*t2, t8=t4*t4;
    if (derivative) *derivative = 1+3*k[0]*t2+5*k[1]*t4+7*k[2]*t6+9*k[3]*t8;
    return t*(1+k[0]*t2+k[1]*t4+k[2]*t6+k[3]*t8);
}
}

String ProjectionModelName(ProjectionModel m) {
    switch(m) {
    case ProjectionModel::Pinhole: return "pinhole";
    case ProjectionModel::BrownConrady: return "brown_conrady";
    case ProjectionModel::Fisheye: return "fisheye";
    }
    return "unknown";
}

ProjectionResult ProjectDirection(const CameraProjection& p, const glm::dvec3& d) {
    if (!std::isfinite(d.x) || !std::isfinite(d.y) || !std::isfinite(d.z) || d.z<=0)
        return {};
    glm::dvec2 v(d.x/d.z,d.y/d.z);
    if (p.model==ProjectionModel::BrownConrady) v=Brown(p,v);
    else if (p.model==ProjectionModel::Fisheye) {
        const f64 r=glm::length(v), t=std::atan(r);
        if (t>p.maxThetaRadians) return {};
        if (r>1e-15) v*=Fish(p,t)/r;
    }
    const glm::dvec2 pixel(p.fx*v.x+p.cx,p.fy*v.y+p.cy);
    return {Finite(pixel),pixel};
}

UnprojectionResult UnprojectPixel(const CameraProjection& p, const glm::dvec2& pixel) {
    if (!Finite(pixel) || !std::isfinite(p.fx) || !std::isfinite(p.fy) || p.fx<=0 || p.fy<=0)
        return {};
    const glm::dvec2 target((pixel.x-p.cx)/p.fx,(pixel.y-p.cy)/p.fy);
    glm::dvec2 v=target;
    if (p.model==ProjectionModel::BrownConrady) {
        bool converged=false;
        for (u32 i=0;i<40;++i) {
            glm::dmat2 j;
            const glm::dvec2 error=Brown(p,v,&j)-target;
            if (glm::length(error*glm::dvec2(p.fx,p.fy))<1e-8) { converged=true; break; }
            const f64 determinant=glm::determinant(j);
            if (!std::isfinite(determinant) || determinant<=1e-12) return {};
            const glm::dvec2 step=glm::inverse(j)*error;
            f64 scale=1;
            while (scale>1.0/1024 && glm::length(Brown(p,v-step*scale)-target)>=glm::length(error))
                scale*=0.5;
            v-=step*scale;
            if (!Finite(v)) return {};
        }
        if (!converged) return {};
    } else if (p.model==ProjectionModel::Fisheye) {
        const f64 r=glm::length(target);
        if (r>Fish(p,p.maxThetaRadians)) return {};
        f64 lo=0,hi=p.maxThetaRadians;
        for (u32 i=0;i<60;++i) {
            const f64 t=(lo+hi)*0.5;
            if (Fish(p,t)<r) lo=t; else hi=t;
        }
        if (r>1e-15) v*=std::tan((lo+hi)*0.5)/r;
    }
    return {true,glm::normalize(glm::dvec3(v,1.0))};
}

Result<void,String> ValidateProjection(const CameraProjection& p,u32 w,u32 h) {
    const auto bad=[](const String& s){ return Result<void,String>::Err(s); };
    if (!w || !h || p.model>ProjectionModel::Fisheye || !std::isfinite(p.fx) ||
        !std::isfinite(p.fy) || p.fx<=0 || p.fy<=0 || !std::isfinite(p.cx) || !std::isfinite(p.cy))
        return bad("invalid camera projection intrinsics or grid");
    for (auto k:p.coefficients) if (!std::isfinite(k)) return bad("non-finite distortion coefficient");
    if (p.model==ProjectionModel::Pinhole)
        for(auto k:p.coefficients) if(k!=0) return bad("pinhole projection cannot carry distortion coefficients");
    if (p.model==ProjectionModel::Fisheye) {
        if (!std::isfinite(p.maxThetaRadians) || p.maxThetaRadians<=0 || p.maxThetaRadians>=halfPi || p.coefficients[4]!=0)
            return bad("fisheye requires four coefficients and a field angle below 90 degrees");
        for (u32 i=0;i<=1024;++i) {
            f64 derivative;
            Fish(p,p.maxThetaRadians*i/1024,&derivative);
            if (!std::isfinite(derivative) || derivative<=1e-8)
                return bad("fisheye projection is not monotone in its valid field");
        }
    }
    // Brown covers the entire sensor. Fisheye can have an invalid border.
    for (u32 y=0;y<=32;++y) for(u32 x=0;x<=32;++x) {
        const glm::dvec2 pixel(w*x/32.0,h*y/32.0);
        const auto ray=UnprojectPixel(p,pixel);
        if (!ray.valid) {
            if(p.model==ProjectionModel::Fisheye) continue;
            return bad("distortion cannot be inverted over the native sensor grid");
        }
        if(p.model==ProjectionModel::BrownConrady) {
            glm::dmat2 j;
            Brown(p,{ray.direction.x/ray.direction.z,ray.direction.y/ray.direction.z},&j);
            if(glm::determinant(j)<=1e-10) return bad("distortion folds over the native sensor grid");
        }
        const auto back=ProjectDirection(p,ray.direction);
        if(!back.valid || glm::length(back.pixel-pixel)>1e-4) return bad("projection round-trip exceeds tolerance");
    }
    return Result<void,String>::Ok();
}

Result<CameraProjection,String> ResolveProjection(const CameraProjection& a,u32 w,u32 h,f64 f,f64 pitch) {
    CameraProjection p=a;
    if(!p.explicitIntrinsics) {
        if(!std::isfinite(f) || !std::isfinite(pitch) || f<=0 || pitch<=0)
            return Result<CameraProjection,String>::Err("invalid focal length or pixel pitch");
        p.fx=p.fy=f*1000/pitch; p.cx=w*0.5; p.cy=h*0.5;
    }
    const auto valid=ValidateProjection(p,w,h);
    if(!valid) return Result<CameraProjection,String>::Err(valid.error());
    return p;
}
} // namespace quantiloom::camera
