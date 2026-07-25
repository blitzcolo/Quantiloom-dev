#include "atmos/AtmosphereBaker.hpp"

#include "core/Log.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace quantiloom {

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kDeg2Rad = kPi / 180.0;
constexpr double kEarthRadiusKm = 6371.0;
constexpr double kGroundRangeMinKm = 0.05;
constexpr double kGroundRangeMaxKm = 20.0;
constexpr double kSlantZenithMinDeg = 110.0;  // cos axis upper bound
constexpr double kSlantZenithMaxDeg = 180.0;
constexpr double kNightSunZenithDeg = 85.0;
constexpr size_t kInferChunkRows = 256;

uint64_t Fnv1a(const void* data, size_t n, uint64_t h) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

// Fill one raw parameter row (manifest feature layout) for a network.
// Weather features are matched by name in the sampled block; the trailing
// 4-column geometry block is filled positionally.
void FillRow(const AtmosNet& net, const AtmosphereNNConfig& cfg,
             double sunZenithDeg, double sunRelAzimuthDeg, double h1Km,
             double viewZenithDeg, double rangeKm, double* row) {
    const auto& names = net.FeatureNames();
    const int P = net.NumFeatures();
    const int sampledEnd = P - 4;
    for (int i = 0; i < sampledEnd; ++i) {
        const std::string& n = names[i];
        double v = 0.0;
        if (n == "atmos_model") v = cfg.atmosModel;
        else if (n == "ihaze") v = cfg.ihaze;
        else if (n == "icld") v = cfg.icld;
        else if (n == "vis_km") v = cfg.visKm;
        else if (n == "rainrt_mm_h") v = cfg.rainrtMmH;
        else if (n == "t_ground_K") v = cfg.tGroundK;
        else if (n == "rh") v = cfg.rh;
        else if (n == "p_hPa") v = cfg.pHPa;
        else if (n == "h2o_scale") v = cfg.h2oScale;
        else if (n == "sun_zenith_deg") v = sunZenithDeg;
        else if (n == "sun_rel_azimuth_deg") v = sunRelAzimuthDeg;
        else if (n == "h1_km") v = h1Km;
        else if (n == "view_zenith_deg") v = viewZenithDeg;
        else if (n == "range_km") v = rangeKm;
        row[i] = v;
    }
    row[P - 4] = h1Km;
    row[P - 3] = (net.PathType() == "horizontal") ? h1Km
                 : (net.PathType() == "sky") ? 100.0 : 0.0;
    row[P - 2] = std::cos(viewZenithDeg * kDeg2Rad);
    row[P - 1] = rangeKm;
}

// Box-average src[K] channels into render lambda bins. binOf[k] gives the
// render bin for NN channel k (or -1 when outside every bin).
void BoxAverage(const std::vector<double>& src, const std::vector<int>& binOf,
                int numBins, double* dst) {
    std::vector<double> sum(numBins, 0.0);
    std::vector<int> cnt(numBins, 0);
    for (size_t k = 0; k < src.size(); ++k) {
        const int b = binOf[k];
        if (b < 0) continue;
        sum[b] += src[k];
        cnt[b] += 1;
    }
    for (int b = 0; b < numBins; ++b)
        dst[b] = cnt[b] > 0 ? sum[b] / cnt[b] : 0.0;
}

}  // namespace

AtmosLambdaGrid RenderBandLambdaGrid(SpectralMode mode, double wavelengthNm) {
    // Sample counts below MUST stay in sync with closesthit.rchit / miss.rmiss.
    auto uniformGrid = [](const char* band, double lo, double hi, int n) {
        AtmosLambdaGrid g;
        g.band = band;
        g.lambdasNm.resize(n);
        for (int i = 0; i < n; ++i)
            g.lambdasNm[i] = lo + (hi - lo) * i / (n - 1);
        return g;
    };
    // Band edges come from GetFusedBandInfo rather than being repeated here.
    // They used to be literals in this switch as well as in Types.hpp and both
    // shaders -- four copies of the same ten numbers, which is how the NIR and
    // SWIR ranges ended up documented wrong elsewhere.
    auto fusedGrid = [&](const char* band, int n) {
        const auto info = GetFusedBandInfo(mode);
        return uniformGrid(band, info->lambdaMinNm, info->lambdaMaxNm, n);
    };
    switch (mode) {
        case SpectralMode::VIS_Fused:
            return fusedGrid("vis", 32);
        case SpectralMode::NIR_Fused:
            return fusedGrid("nir", 16);
        case SpectralMode::SWIR_Fused:
            return fusedGrid("swir", 16);
        case SpectralMode::MWIR_Fused:
            return fusedGrid("mwir", 16);
        case SpectralMode::LWIR_Fused:
            return fusedGrid("lwir", 16);
        case SpectralMode::RGB: {
            AtmosLambdaGrid g;
            g.band = "vis";
            g.lambdasNm = {650.0, 550.0, 450.0};  // iLambda 0/1/2 = R/G/B
            g.windowHalfWidthNm = 5.0;
            return g;
        }
        case SpectralMode::Single: {
            // These are the NN atmosphere's *trained coverage* per band, not the
            // fused-mode integration ranges above, and deliberately differ: vis
            // reaches 800 nm here versus 780 for VIS_Fused. Do not "unify" them
            // with GetFusedBandInfo -- a single wavelength is admissible anywhere
            // the network was trained, which is a wider question than which band
            // a fused render integrates.
            AtmosLambdaGrid g;
            struct Range { const char* band; double lo, hi; };
            static const Range kRanges[] = {{"vis", 400.0, 800.0},
                                            {"nir", 930.0, 1200.0},
                                            {"swir", 1400.0, 2400.0},
                                            {"mwir", 3000.0, 5000.0},
                                            {"lwir", 8000.0, 12000.0}};
            for (const Range& r : kRanges) {
                if (wavelengthNm >= r.lo && wavelengthNm <= r.hi) {
                    g.band = r.band;
                    g.lambdasNm = {wavelengthNm};
                    g.windowHalfWidthNm = 2.0;
                    return g;
                }
            }
            g.error = "wavelength " + std::to_string(wavelengthNm) +
                      " nm is outside NN atmosphere coverage; supported ranges "
                      "(nm): vis [400, 800], nir [930, 1200], swir [1400, 2400], "
                      "mwir [3000, 5000], lwir [8000, 12000]";
            return g;
        }
        default:
            return {};  // Multispectral etc.: no NN atmosphere
    }
}

double AtmosphereBaker::SlantRangeKm(double h1Km, double viewZenithDeg) {
    const double r1 = kEarthRadiusKm + h1Km;
    const double thetaMin =
        180.0 - std::asin(kEarthRadiusKm / r1) / kDeg2Rad;
    const double theta = std::max(viewZenithDeg, thetaMin + 0.05);
    const double mu = std::cos(theta * kDeg2Rad);
    const double disc =
        r1 * r1 * mu * mu - (r1 * r1 - kEarthRadiusKm * kEarthRadiusKm);
    return -r1 * mu - std::sqrt(std::max(disc, 0.0));
}

uint64_t AtmosphereBaker::BakeKey(const AtmosphereNNConfig& cfg,
                                  const std::string& band,
                                  const std::vector<double>& lambdasNm) {
    uint64_t h = cfg.BakeHash();
    h = Fnv1a(band.data(), band.size(), h);
    h = Fnv1a(lambdasNm.data(), lambdasNm.size() * sizeof(double), h);
    return h;
}

AtmosBakeResult AtmosphereBaker::Bake(const AtmosphereNNConfig& cfg,
                                      const std::string& band,
                                      const std::vector<double>& lambdasNm,
                                      double windowHalfWidthNm) {
    const auto t0 = std::chrono::steady_clock::now();
    if (lambdasNm.empty())
        throw std::runtime_error("AtmosphereBaker: empty lambda grid");

    // ---------------------------------------------------------- geometry --
    double h1 = cfg.h1Km;
    if (h1 > 12.0) {
        QL_LOG_WARN("AtmosphereBaker: h1 = {:.2f} km above slant training "
                    "domain, clamped to 12 km", h1);
        h1 = 12.0;
    }
    const bool ground = h1 < 0.5;
    const std::string geom = ground ? "ground" : "slant";

    const bool night = cfg.sunZenithDeg > kNightSunZenithDeg;
    const double sunZen = std::min(cfg.sunZenithDeg, kNightSunZenithDeg);
    if (night)
        QL_LOG_WARN("AtmosphereBaker: sun zenith {:.1f} deg > 85 (night); "
                    "solar-scatter terms zeroed, reflective-band inference "
                    "clamped to 85 deg", cfg.sunZenithDeg);

    // ------------------------------------------------------------- nets ---
    const bool thermal = (band == "mwir" || band == "lwir");
    std::vector<std::array<std::string, 3>> required = {
        {band, geom, "tau"}, {band, geom, "lpath"}};
    if (thermal) required.push_back({band, geom, "ldown"});
    pack_.RequireNets(required);
    const AtmosNet& tauNet = pack_.Get(band, geom, "tau");
    const AtmosNet& lpathNet = pack_.Get(band, geom, "lpath");
    const AtmosNet* ldownNet = thermal ? &pack_.Get(band, geom, "ldown") : nullptr;

    // Render lambda samples must lie inside the network band grid. Tolerance
    // of two channel widths absorbs sub-nm mismatches at the band edges
    // (e.g. MWIR loop starts at 3000.0 nm, grid tops out at 3000.3 nm).
    const double bandLoNm = tauNet.Band().MinWavelengthNm();
    const double bandHiNm = tauNet.Band().MaxWavelengthNm();
    for (double lam : lambdasNm) {
        const double tolNm = 2.0 * tauNet.Band().dv_cm * lam * lam / 1e7;
        if (lam < bandLoNm - tolNm || lam > bandHiNm + tolNm)
            throw std::runtime_error(
                "AtmosphereBaker: render wavelength " + std::to_string(lam) +
                " nm outside " + band + " network coverage [" +
                std::to_string(bandLoNm) + ", " + std::to_string(bandHiNm) +
                "] nm");
    }

    // ------------------------------------------------------------- axes ---
    const int countA = std::clamp(cfg.lutASamples, 2, 256);
    double aStart, aStep;
    if (ground) {
        aStart = std::log(kGroundRangeMinKm);
        aStep = (std::log(kGroundRangeMaxKm) - aStart) / (countA - 1);
    } else {
        aStart = std::cos(kSlantZenithMaxDeg * kDeg2Rad);  // -1
        aStep = (std::cos(kSlantZenithMinDeg * kDeg2Rad) - aStart) / (countA - 1);
    }
    const bool lpathHasAz = lpathNet.HasSunFeatures();
    const int countAz = lpathHasAz ? std::clamp(cfg.lutAzSamples, 2, 64) : 1;
    const double azStart = 0.0;
    const double azStep = countAz > 1 ? 180.0 / (countAz - 1) : 0.0;

    // Per-a geometry (shared by tau and lpath rows)
    std::vector<double> viewZenith(countA), rangeKm(countA);
    for (int ia = 0; ia < countA; ++ia) {
        const double a = aStart + aStep * ia;
        if (ground) {
            viewZenith[ia] = 90.0;
            rangeKm[ia] = std::exp(a);
        } else {
            viewZenith[ia] = std::acos(std::clamp(a, -1.0, 1.0)) / kDeg2Rad;
            rangeKm[ia] = SlantRangeKm(h1, viewZenith[ia]);
        }
    }

    // ------------------------------------------------- lambda bin mapping --
    // NN channel k (wavenumber grid) -> render bin. Bin edges are midpoints
    // between consecutive render samples.
    const int numLambda = static_cast<int>(lambdasNm.size());
    auto makeBinOf = [&](const AtmosBandGrid& grid) {
        std::vector<int> binOf(grid.K, -1);
        if (windowHalfWidthNm > 0.0) {
            // Discrete-lambda mode: nearest sample within the window
            for (int k = 0; k < grid.K; ++k) {
                const double lam = grid.WavelengthNmAt(k);
                int best = -1;
                double bestDist = windowHalfWidthNm;
                for (int i = 0; i < numLambda; ++i) {
                    const double d = std::abs(lam - lambdasNm[i]);
                    if (d <= bestDist) {
                        bestDist = d;
                        best = i;
                    }
                }
                binOf[k] = best;
            }
            return binOf;
        }
        std::vector<double> edges(numLambda + 1);
        edges[0] = std::max(lambdasNm[0] - 0.5 * (numLambda > 1
                       ? (lambdasNm[1] - lambdasNm[0]) : grid.MaxWavelengthNm() - grid.MinWavelengthNm()),
                       bandLoNm);
        for (int i = 1; i < numLambda; ++i)
            edges[i] = 0.5 * (lambdasNm[i - 1] + lambdasNm[i]);
        edges[numLambda] = std::min(
            lambdasNm[numLambda - 1] + 0.5 * (numLambda > 1
                ? (lambdasNm[numLambda - 1] - lambdasNm[numLambda - 2])
                : grid.MaxWavelengthNm() - grid.MinWavelengthNm()),
            bandHiNm);
        for (int k = 0; k < grid.K; ++k) {
            const double lam = grid.WavelengthNmAt(k);
            if (lam < edges[0] || lam > edges[numLambda]) continue;
            // Render samples ascend in nm; binary search the edge array.
            const int b = static_cast<int>(
                std::upper_bound(edges.begin(), edges.end(), lam) -
                edges.begin()) - 1;
            if (b >= 0 && b < numLambda) binOf[k] = b;
        }
        return binOf;
    };

    // ---------------------------------------------------------- layout ----
    AtmosBakeResult result;
    AtmosNNHeaderGPU& hd = result.header;
    hd.enabled = 1;
    hd.pathMode = ground ? 0u : 1u;
    hd.numLambda = static_cast<uint32_t>(numLambda);
    hd.countA = static_cast<uint32_t>(countA);
    hd.countAz = static_cast<uint32_t>(countAz);
    hd.hasLdown = thermal ? 1u : 0u;
    hd.hasSky = 0;
    hd.thermalBand = thermal ? 1u : 0u;
    hd.aStart = static_cast<float>(aStart);
    hd.aStep = static_cast<float>(aStep);
    hd.azStart = static_cast<float>(azStart);
    hd.azStep = static_cast<float>(azStep);
    hd.tauOffset = 0;
    hd.lpathOffset = hd.tauOffset + static_cast<uint32_t>(numLambda * countA);
    hd.ldownOffset =
        hd.lpathOffset + static_cast<uint32_t>(numLambda * countA * countAz);
    hd.skyOffset = hd.ldownOffset + static_cast<uint32_t>(thermal ? numLambda : 0);
    result.data.assign(hd.skyOffset, 0.0f);

    // ------------------------------------------------------------- tau ----
    {
        const AtmosNet& net = tauNet;
        const int P = net.NumFeatures();
        const int K = net.K();
        const std::vector<int> binOf = makeBinOf(net.Band());
        std::vector<double> rows(static_cast<size_t>(countA) * P);
        for (int ia = 0; ia < countA; ++ia)
            FillRow(net, cfg, sunZen, 0.0, h1, viewZenith[ia], rangeKm[ia],
                    rows.data() + static_cast<size_t>(ia) * P);
        std::vector<double> out(static_cast<size_t>(countA) * K);
        for (size_t s = 0; s < static_cast<size_t>(countA); s += kInferChunkRows) {
            const size_t n = std::min(kInferChunkRows, static_cast<size_t>(countA) - s);
            net.Infer(rows.data() + s * P, n, out.data() + s * K);
        }
        std::vector<double> tauK(K), avg(numLambda);
        for (int ia = 0; ia < countA; ++ia) {
            const double* delta = out.data() + static_cast<size_t>(ia) * K;
            if (net.IsOpaque(delta)) {
                for (int i = 0; i < numLambda; ++i)
                    result.data[hd.tauOffset + static_cast<size_t>(i) * countA + ia] = 0.0f;
                continue;
            }
            for (int k = 0; k < K; ++k) tauK[k] = std::exp(-delta[k]);
            BoxAverage(tauK, binOf, numLambda, avg.data());
            for (int i = 0; i < numLambda; ++i)
                result.data[hd.tauOffset + static_cast<size_t>(i) * countA + ia] =
                    static_cast<float>(avg[i]);
        }
    }

    // ------------------------------------------------------------ lpath ---
    {
        const AtmosNet& net = lpathNet;
        const int P = net.NumFeatures();
        const int K = net.K();
        const int T = net.T();  // 1, or 2 for mwir (PTH_THRML + SOL_SCAT)
        const std::vector<int> binOf = makeBinOf(net.Band());
        const size_t numRows = static_cast<size_t>(countA) * countAz;
        std::vector<double> rows(numRows * P);
        for (int ia = 0; ia < countA; ++ia) {
            for (int iaz = 0; iaz < countAz; ++iaz) {
                const double relAz = azStart + azStep * iaz;
                FillRow(net, cfg, sunZen, relAz, h1, viewZenith[ia], rangeKm[ia],
                        rows.data() + (static_cast<size_t>(ia) * countAz + iaz) * P);
            }
        }
        std::vector<double> lvK(K), avg(numLambda);
        std::vector<double> out(kInferChunkRows * static_cast<size_t>(T) * K);
        for (size_t s = 0; s < numRows; s += kInferChunkRows) {
            const size_t n = std::min(kInferChunkRows, numRows - s);
            net.Infer(rows.data() + s * P, n, out.data());
            for (size_t r = 0; r < n; ++r) {
                const size_t row = s + r;
                const double* thrml = out.data() + (r * T + 0) * K;
                const double* sol = T > 1 ? out.data() + (r * T + 1) * K : nullptr;
                for (int k = 0; k < K; ++k) {
                    double lv = thrml[k];
                    if (sol && !night) lv += sol[k];
                    // W cm^-2 sr^-1 / cm^-1 -> W m^-2 sr^-1 nm^-1
                    const double lam = net.Band().WavelengthNmAt(k);
                    lvK[k] = lv * 1e11 / (lam * lam);
                }
                BoxAverage(lvK, binOf, numLambda, avg.data());
                const int ia = static_cast<int>(row / countAz);
                const int iaz = static_cast<int>(row % countAz);
                for (int i = 0; i < numLambda; ++i)
                    result.data[hd.lpathOffset +
                                (static_cast<size_t>(i) * countA + ia) * countAz +
                                iaz] = static_cast<float>(avg[i]);
            }
        }
    }

    // ------------------------------------------------------------ ldown ---
    if (thermal) {
        const AtmosNet& net = *ldownNet;
        const int P = net.NumFeatures();
        const int K = net.K();
        const std::vector<int> binOf = makeBinOf(net.Band());
        std::vector<double> row(P);
        // ldown ignores viewing geometry (excluded inputs); fill neutral values.
        FillRow(net, cfg, sunZen, 0.0, h1, ground ? 90.0 : 135.0,
                ground ? 1.0 : SlantRangeKm(h1, 135.0), row.data());
        std::vector<double> out(static_cast<size_t>(net.T()) * K);
        net.Infer(row.data(), 1, out.data());
        std::vector<double> lvK(K), avg(numLambda);
        for (int k = 0; k < K; ++k) {
            const double lam = net.Band().WavelengthNmAt(k);
            lvK[k] = out[k] * 1e11 / (lam * lam);
        }
        BoxAverage(lvK, binOf, numLambda, avg.data());
        for (int i = 0; i < numLambda; ++i)
            result.data[hd.ldownOffset + i] = static_cast<float>(avg[i]);
    }

    const auto t1 = std::chrono::steady_clock::now();
    const double ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    QL_LOG_INFO("AtmosphereBaker: baked {} {} LUT in {:.0f} ms "
                "({} lambda x {} a x {} az, {} floats)",
                band, geom, ms, numLambda, countA, countAz, result.data.size());
    return result;
}

}  // namespace quantiloom
