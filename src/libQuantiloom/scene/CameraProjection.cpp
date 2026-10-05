#include "scene/CameraProjection.hpp"
#include <cmath>
#include <limits>
#include <vector>
#include <algorithm>
#include <functional>

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
// Isolate roots using derivative roots: each intervening interval is monotone.
std::vector<f64> PolynomialRoots(std::vector<f64> c,f64 lo,f64 hi) {
    while(c.size()>1 && c.back()==0) c.pop_back();
    if(c.size()<2) return {};
    const auto eval=[&](f64 x){f64 v=0;for(auto i=c.rbegin();i!=c.rend();++i)v=v*x+*i;return v;};
    std::vector<f64> derivative;
    for(size_t i=1;i<c.size();++i)derivative.push_back(i*c[i]);
    auto cuts=PolynomialRoots(derivative,lo,hi);cuts.insert(cuts.begin(),lo);cuts.push_back(hi);
    std::vector<f64> roots;
    for(auto x:cuts)if(std::abs(eval(x))<1e-12)roots.push_back(x);
    for(size_t i=1;i<cuts.size();++i) {
        f64 a=cuts[i-1],b=cuts[i],fa=eval(a);
        if((fa>0)==(eval(b)>0))continue;
        for(u32 j=0;j<80;++j){const f64 m=(a+b)*.5;if((eval(m)>0)==(fa>0))a=m;else b=m;}
        roots.push_back((a+b)*.5);
    }
    std::sort(roots.begin(),roots.end());
    roots.erase(std::unique(roots.begin(),roots.end(),[](f64 a,f64 b){return std::abs(a-b)<1e-12;}),roots.end());
    return roots;
}
struct Interval {f64 lo,hi;};
Interval operator+(Interval a,Interval b){return {a.lo+b.lo,a.hi+b.hi};}
Interval operator*(Interval a,Interval b){const f64 v[]={a.lo*b.lo,a.lo*b.hi,a.hi*b.lo,a.hi*b.hi};return {*std::min_element(v,v+4),*std::max_element(v,v+4)};}
Interval I(f64 v){return {v,v};}
Interval Square(Interval a){return {a.lo<=0 && a.hi>=0 ? 0 : std::min(a.lo*a.lo,a.hi*a.hi),std::max(a.lo*a.lo,a.hi*a.hi)};}
bool CertifyBrown(const CameraProjection& p,Interval x,Interval y,u32 depth) {
    const auto& k=p.coefficients;const auto x2=Square(x),y2=Square(y),r=x2+y2;
    const auto a=I(1)+I(k[0])*r+I(k[1])*Square(r)+I(k[4])*Square(r)*r;
    const auto da=I(2)*(I(k[0])+I(2*k[1])*r+I(3*k[4])*Square(r));
    const auto jx=a+x2*da+I(2*k[2])*y+I(6*k[3])*x;
    const auto jy=a+y2*da+I(6*k[2])*y+I(2*k[3])*x;
    const auto cross=x*y*da+I(2*k[2])*x+I(2*k[3])*y;
    const auto product=jx*jy;const auto c2=Square(cross);
    if(jx.lo>1e-10 && jy.lo>1e-10 && product.lo-c2.hi>1e-10)return true;
    if(depth==0 || jx.hi<=0 || jy.hi<=0)return false;
    if(x.hi-x.lo>=y.hi-y.lo){const auto m=(x.lo+x.hi)*.5;return CertifyBrown(p,{x.lo,m},y,depth-1)&&CertifyBrown(p,{m,x.hi},y,depth-1);}
    const auto m=(y.lo+y.hi)*.5;return CertifyBrown(p,x,{y.lo,m},depth-1)&&CertifyBrown(p,x,{m,y.hi},depth-1);
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

LensResultV2 UnprojectPixelV2(const CameraProjection& p,const glm::dvec2& pixel,u32 w,u32 h) {
    if(!Finite(pixel))return {};
    if(pixel.x<0 || pixel.y<0 || pixel.x>=w || pixel.y>=h)return {LensValidity::OutsideImage,{}};
    if(p.model==ProjectionModel::Fisheye && p.fx>0 && p.fy>0 &&
       glm::length(glm::dvec2((pixel.x-p.cx)/p.fx,(pixel.y-p.cy)/p.fy))>Fish(p,p.maxThetaRadians))
        return {LensValidity::OutsideField,{}};
    const auto ray=UnprojectPixel(p,pixel);
    return {ray.valid ? LensValidity::Valid : LensValidity::InverseFailed,ray.direction};
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
        auto extrema=PolynomialRoots({3*p.coefficients[0],10*p.coefficients[1],
            21*p.coefficients[2],36*p.coefficients[3]},0,p.maxThetaRadians*p.maxThetaRadians);
        extrema.push_back(0);extrema.push_back(p.maxThetaRadians*p.maxThetaRadians);
        for (auto squaredTheta:extrema) {
            f64 derivative;
            Fish(p,std::sqrt(squaredTheta),&derivative);
            if (!std::isfinite(derivative) || derivative<=1e-8)
                return bad("fisheye projection is not monotone in its valid field");
        }
    }
    // Brown covers the entire sensor. Fisheye can have an invalid border.
    Interval bx{0,0},by{0,0};
    for (u32 y=0;y<=32;++y) for(u32 x=0;x<=32;++x) {
        const glm::dvec2 pixel(w*x/32.0,h*y/32.0);
        const auto ray=UnprojectPixel(p,pixel);
        if (!ray.valid) {
            if(p.model==ProjectionModel::Fisheye) continue;
            return bad("distortion cannot be inverted over the native sensor grid");
        }
        if(p.model==ProjectionModel::BrownConrady) {
            const f64 rx=ray.direction.x/ray.direction.z,ry=ray.direction.y/ray.direction.z;
            bx.lo=std::min(bx.lo,rx);bx.hi=std::max(bx.hi,rx);
            by.lo=std::min(by.lo,ry);by.hi=std::max(by.hi,ry);
            glm::dmat2 j;
            Brown(p,{ray.direction.x/ray.direction.z,ray.direction.y/ray.direction.z},&j);
            if(glm::determinant(j)<=1e-10) return bad("distortion folds over the native sensor grid");
        }
        const auto back=ProjectDirection(p,ray.direction);
        if(!back.valid || glm::length(back.pixel-pixel)>1e-4) return bad("projection round-trip exceeds tolerance");
    }
    if(p.model==ProjectionModel::BrownConrady) {
        const f64 pad=1e-4*std::max({1.0,bx.hi-bx.lo,by.hi-by.lo});
        if(!CertifyBrown(p,{bx.lo-pad,bx.hi+pad},{by.lo-pad,by.hi+pad},16))
            return bad("Brown distortion invertibility cannot be certified over its lens domain");
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
