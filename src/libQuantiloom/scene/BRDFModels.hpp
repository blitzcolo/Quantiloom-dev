#pragma once
#include <cmath>
#include <complex>
#include <algorithm>

// All angles in radians. Returns BRDF value in sr^-1.
// Bugs fixed vs. original source (see Physically_based_BRDF_report.md).

namespace quantiloom {

static constexpr double BRDF_PI = 3.14159265358979323846;

// 1. Lambertian (BRDFsea)
inline double brdf_lambertian(double rho,
    double /*theta_i*/, double /*phi_i*/, double /*theta_s*/, double /*phi_s*/)
{
    return rho / BRDF_PI;
}

// 2. Five-parameter hybrid (BRDF5P)
inline double brdf_5p(double kb, double kr, double b, double a, double kd,
    double theta_i, double phi_i, double theta_s, double phi_s)
{
    double cos_i = std::cos(theta_i), sin_i = std::sin(theta_i);
    double cos_s = std::cos(theta_s), sin_s = std::sin(theta_s);
    double cosr2 = 0.5 * (cos_i*cos_s - sin_i*sin_s*std::cos(phi_s - phi_i) + 1.0);
    double cosr  = std::sqrt(cosr2);
    double cosa  = (cos_i + cos_s) / (2.0 * cosr);
    double ra    = 1.0 - cosr;
    if (kb == 0.0)
        return kd / BRDF_PI;
    return kb * kr*kr * cosa * std::exp(b * std::pow(ra, a))
           / (1.0 + (kr*kr - 1.0) * cosa) / cos_s + kd;
}

// 3. Kernel-driven RossThick-LiTransit (BRDFKD)
inline double brdf_kernel_driven(double fiso, double fgeo, double fvol,
    double theta_i, double phi_i, double theta_s, double phi_s)
{
    auto sec = [](double x){ return 1.0 / std::cos(x); };

    double D2 = std::tan(theta_i)*std::tan(theta_i) + std::tan(theta_s)*std::tan(theta_s)
              - 2.0*std::tan(theta_i)*std::tan(theta_s)*std::cos(phi_s - phi_i);
    double cost = 2.0 * std::sqrt(D2 + std::pow(std::tan(theta_i)*std::tan(theta_s)*std::sin(phi_s - phi_i), 2))
                / (sec(theta_i) + sec(theta_s));
    cost = std::max(-1.0, std::min(1.0, cost));

    double t    = std::acos(cost);
    double O    = (t - std::sin(t)*cost) * (sec(theta_i) + sec(theta_s)) / BRDF_PI;
    double B    = sec(theta_i) + sec(theta_s) - O;
    double cosr = std::cos(theta_i)*std::cos(theta_s)
                + std::sin(theta_i)*std::sin(theta_s)*std::cos(phi_s - phi_i);
    double Ksparse = O - sec(theta_i) - sec(theta_s) + 0.5*(1.0+cosr)*sec(theta_s);
    double Kgeo = (B > 2.0) ? 2.0*Ksparse/B : Ksparse;
    double Kvol = ((0.5*BRDF_PI - std::acos(cosr))*cosr + std::sin(std::acos(cosr)))
                / (std::cos(theta_i) + std::cos(theta_s)) - BRDF_PI*0.25;
    return fiso + fgeo*Kgeo + fvol*Kvol;
}

// 4. Cox-Munk ocean (BRDFocean) — bug fixed: yn uses Z_y, not Z_x
namespace detail {
    inline double cox_munk(double Z_x, double Z_y, double psi, double U)
    {
        double zx = Z_x*std::cos(psi) + Z_y*std::sin(psi);
        double zy = -Z_x*std::sin(psi) + Z_y*std::cos(psi);
        double sx2 = 0.00316*U, sy2 = 0.003 + 0.00192*U;
        double sx = std::sqrt(sx2), sy = std::sqrt(sy2);
        double C21=0.01-0.0086*U, C03=0.04-0.033*U, C40=0.40, C22=0.12, C04=0.23;
        double X = zx/sx, Y = zy/sy;
        double p0  = std::exp(-0.5*(zx*zx/sx2 + zy*zy/sy2)) / (2.0*BRDF_PI*sx*sy);
        double p01 = 1.0 - 0.5*C21*(Y*Y-1)*X - C03/6.0*(X*X*X-3*X);
        double p02 = C22/4.0*(Y*Y-1)*(X*X-1) + C40/24.0*(Y*Y*Y*Y-6*Y*Y+3);
        double p03 = C04/24.0*(X*X*X*X-6*X*X+3);
        return p0*(p01+p02+p03);
    }

    inline double smith_shadow(double theta, double phi, double U)
    {
        double sx2 = 0.00316*U, sy2 = 0.003 + 0.00192*U;
        double s2  = sx2*std::cos(phi)*std::cos(phi) + sy2*std::sin(phi)*std::sin(phi);
        double s   = std::sqrt(s2);
        double cot = 1.0 / std::tan(theta + 1e-5);
        double v   = cot / (std::sqrt(2.0)*s);
        double s01 = std::exp(-v*v) - v*std::sqrt(BRDF_PI)*(1.0-std::erf(v));
        return s01 / (2.0*v*std::sqrt(BRDF_PI));
    }

    inline double fresnel_unpolarized(double cos_w, double n)
    {
        double sin_w  = std::sqrt(1.0 - cos_w*cos_w);
        double sin_w2 = sin_w / n;
        double cos_w2 = std::sqrt(1.0 - sin_w2*sin_w2);
        double rv = (n*cos_w - cos_w2) / (n*cos_w + cos_w2);
        double rp = (cos_w - n*cos_w2) / (cos_w + n*cos_w2);
        return 0.5*(rv*rv + rp*rp);
    }
} // namespace detail

inline double brdf_ocean(double n, double wind_speed, double wind_dir,
    double theta_i, double phi_i, double theta_s, double phi_s)
{
    double xr = std::sin(theta_i)*std::cos(phi_i), yr = std::sin(theta_i)*std::sin(phi_i), zr = std::cos(theta_i);
    double xs = std::sin(theta_s)*std::cos(phi_s), ys = std::sin(theta_s)*std::sin(phi_s), zs = std::cos(theta_s);
    double Z_x = -(xr+xs)/(zr+zs), Z_y = -(yr+ys)/(zr+zs);
    double m = std::sqrt(1.0 + Z_x*Z_x + Z_y*Z_y);
    double xn = -Z_x/m, yn = -Z_y/m, zn = 1.0/m;  // fixed: yn uses Z_y

    double p       = detail::cox_munk(Z_x, Z_y, wind_dir, wind_speed);
    double W       = xn*xr + yn*yr + zn*zr;
    int    H       = W > 0 ? 1 : 0;
    double Sr      = detail::smith_shadow(theta_i, phi_i, wind_speed);
    double Ss      = detail::smith_shadow(theta_s, phi_s, wind_speed);
    double cos_omg = W;
    double rho     = detail::fresnel_unpolarized(cos_omg, n);
    double num     = BRDF_PI * rho * p * W * H;
    double den     = 4.0 * zn*zn*zn * cos_omg * zs * (1.0+Sr+Ss);
    return num / den;
}

// 5. Staylor-Suttles vegetation hotspot
inline double brdf_staylor_suttles(double c1, double c2, double c3, double N,
    double theta_i, double phi_i, double theta_s, double phi_s)
{
    double ui=std::cos(theta_i), ur=std::cos(theta_s);
    double si=std::sin(theta_i), sr=std::sin(theta_s);
    double B = (c1 + c2*std::pow(ui*ur/(ui+ur), N)) / (ui*ur);
    double num = 1.0 + c3*std::pow(ui*ur - si*sr*std::cos(phi_s - phi_i), 2);
    double den = 1.0 + c3*(ui*ui*ur*ur + 0.5*si*si*sr*sr);
    return B * num / den;
}

// 6. Otterman rough surface — constructor-in-private bug fixed (now a free function)
inline double brdf_otterman(double B, double T, double L, double s, double rp,
    double theta_i, double phi_i, double theta_s, double phi_s)
{
    (void)phi_i;
    double cot_i = 1.0/std::tan(theta_i), cot_s = 1.0/std::tan(theta_s);
    double Rg = 0.25*(B*(std::sin(phi_s)-phi_s*std::cos(phi_s))
                    + T*(std::sin(phi_s)-(phi_s-BRDF_PI)*std::cos(phi_s)))
              / (cot_i+cot_s) + L;
    double Rp_ = 0.25*rp*(std::sin(phi_s)-phi_s*std::cos(phi_s)) / (cot_i+cot_s);
    return (Rg-Rp_)*std::exp(-s*(std::tan(theta_i)+std::tan(theta_s))) + Rp_;
}

} // namespace quantiloom
