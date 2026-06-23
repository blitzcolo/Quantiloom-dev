#pragma once
#include "ModtranRunner.hpp"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace qltrans {

static const int FILENAMELENGHTH = 500;

struct MODATA { float wvLenth; float data; };

class ModtranCalc {
public:
    // wave: 1=VIS 2=MWIR 3=LWIR (used for ICLD selection only)
    // wl_start/wl_stop/wl_step: custom wavelength range in nm (0 = use wave defaults)
    // sun_zen_start/stop/step: solar zenith angle range in degrees
    ModtranCalc(int wave, int weather, int atmos, int ihaze,
                double wl_start, double wl_stop, double wl_step,
                double sun_zen_start = 0, double sun_zen_stop = 90, double sun_zen_step = 1)
        : m_wave(wave), m_weather(weather), m_atmos(atmos), m_ihaze(ihaze),
          m_wl_start(wl_start), m_wl_stop(wl_stop), m_wl_step(wl_step),
          m_sun_zen_start(sun_zen_start), m_sun_zen_stop(sun_zen_stop),
          m_sun_zen_step(sun_zen_step)
    {
        // Fill defaults from wave if not specified
        if (m_wl_start <= 0 || m_wl_stop <= 0) {
            double b0, b1;
            switch (wave) { case 1: b0=380; b1=760; break; case 2: b0=3000; b1=5000; break; default: b0=8000; b1=12000; }
            if (m_wl_start <= 0) m_wl_start = b0;
            if (m_wl_stop  <= 0) m_wl_stop  = b1;
        }
        m_n_sun = (int)((m_sun_zen_stop - m_sun_zen_start) / m_sun_zen_step) + 1;
    }

    void run(const std::filesystem::path& workDir,
             const char* outsunFile, const char* outskyFile, const char* outtransFile)
    {
        inputmake(workDir);

        tp5sun(workDir);
        RunModtran(workDir);
        saveSunData(workDir, outsunFile);

        tp5sky(workDir);
        RunModtran(workDir);
        saveSkyData(workDir, outskyFile);

        tp5trans(workDir);
        RunModtran(workDir);
        savetransData(workDir, outtransFile);
    }

    int sunAngleCount() const { return m_n_sun; }

private:
    int m_wave, m_weather, m_atmos, m_ihaze;
    double m_wl_start, m_wl_stop, m_wl_step;
    double m_sun_zen_start, m_sun_zen_stop, m_sun_zen_step;
    int m_n_sun;

    static std::string wd(const std::filesystem::path& d, const std::string& f) {
        return (d / f).string();
    }

    void inputmake(const std::filesystem::path& dir) {
        FILE* f = fopen(wd(dir, "AtmInput.txt").c_str(), "w");
        fprintf(f, "Wavelength:%d\n", m_wave);
        fprintf(f, "Atmosphere model: %d\n", m_atmos);
        fprintf(f, "Weather: %d\n", m_weather);
        fprintf(f, "IHAZE: %d\n", m_ihaze);
        fprintf(f, "Solar zenith: %.1f-%.1f step %.1f (%d points)\n",
                m_sun_zen_start, m_sun_zen_stop, m_sun_zen_step, m_n_sun);
        fclose(f);
    }

    static int WvNum(const char* file) {
        int num = 0; float a, b;
        FILE* f = fopen(file, "r"); if (!f) return 0;
        fscanf(f, "%f%*f", &a);
        while (!feof(f)) { fscanf(f, "%f%*f", &b); num++; }
        fclose(f); return num;
    }

    static int Minmin(const float* data, float val, int len) {
        float best = 1e6f; int idx = 0;
        for (int i = 0; i < len; i++) { float d = fabsf(data[i]-val); if (d < best) { best=d; idx=i; } }
        return idx;
    }

    static void GetIndex(int* index, const char* file, float* data, int n) {
        int wn = WvNum(file);
        FILE* f = fopen(file, "r");
        std::vector<float> wv(wn);
        for (int i = 0; i < wn; i++) fscanf(f, "%f %*f", &wv[i]);
        fclose(f);
        for (int i = 0; i < n; i++) index[i] = Minmin(wv.data(), data[i], wn);
    }

    static void Getdat(int* index, const char* file, float* data, int n) {
        int wn = WvNum(file);
        FILE* f = fopen(file, "r");
        std::vector<float> rd(wn);
        for (int i = 0; i < wn; i++) fscanf(f, "%*f %f", &rd[i]);
        fclose(f);
        for (int i = 0; i < n; i++) data[i] = rd[index[i]];
    }

    static void ModImport(std::vector<MODATA>& modData, const char* file) {
        FILE* f = fopen(file, "r"); if (!f) return;
        for (auto& m : modData) fscanf(f, "%f%f", &m.wvLenth, &m.data);
        fclose(f);
    }

    void GetModDataSolar(std::vector<MODATA>& modData, const char* file, int* index, int n) const {
        int wn = WvNum(file) / m_n_sun;
        std::vector<MODATA> tmp(m_n_sun * wn);
        ModImport(tmp, file);
        for (int i = 0; i < m_n_sun; i++)
            for (int j = 0; j < n; j++) {
                modData[i*n+j].wvLenth = tmp[i*wn+index[j]].wvLenth;
                modData[i*n+j].data    = tmp[i*wn+index[j]].data;
            }
    }

    static void writeCard2(FILE* tp5, int MODEL, int IEMSCT, int TYPE, int IHAZE, int ICLD) {
        fprintf(tp5, "%c%c%3d%5d%5d%5d%5d%5d%5d%5d%5d%5d%5d%5d%5d%8.3f%7.2f\n",
                'T',' ',MODEL,TYPE,IEMSCT,0,0,0,0,0,0,0,0,0,0,0.00,0.0);
        fprintf(tp5, "%c%c%3d%c%4d%10.3f%s\n",'F','T',2,'F',5,330.0,"                                                     ");
        fprintf(tp5, "%s%3d%c%4d%s%2d%5d%5d%5d%10.3f%10.3f%10.3f%10.3f%10.3f\n",
                "  ",IHAZE,' ',0,"   ",0,0,ICLD,0,0.0,0.0,0.0,0.0,0.0);
        if (ICLD>0&&ICLD<6)
            fprintf(tp5,"%8.3f%8.3f%8.3f%4d%4d%8.3f%8.3f%8.3f%8.3f%8.3f%8.3f\n",
                    -9.0,-9.0,-9.0,-9,-9,-9.0,-9.0,-9.0,-9.0,-9.0,-9.0);
        if (ICLD==18)
            fprintf(tp5,"%8.3f%8.3f%8.3f%s\n",-9.0,-9.0,-9.0,"         0");
    }

    static int getICLD(int weather) { if (weather==0) return 0; return 5; }

    void getBands(double& w1, double& w2) const {
        // MODTRAN uses wavenumber (cm^-1): w = 10000/lambda_um = 10000000/lambda_nm
        w1 = 10000000.0 / m_wl_stop;
        w2 = 10000000.0 / m_wl_start;
    }

    // Write Card 3 for IEMSCT=2 (sky radiance / transmittance).
    // Uses IPARM=1 (lat/lon/time -> solar position) with an equator/equinox trick:
    //   lat=0, lon=0, IDAY=80 (spring equinox, solar declination ≈ 0)
    //   At equator on equinox: θ_sun = hour_angle = (UTC - 12) × 15°
    //   So UTC = 12 + θ_sun / 15 gives exact solar zenith angle.
    // Observer fixed at nadir (obs_zenith=0).
    static void writeCard3_radiance(FILE* tp5, double obs_zenith, double sun_zenith) {
        // Card 3: H1=0, H2=0, ANGLE=obs_zenith
        fprintf(tp5, "%10.3f%10.3f%10.3f%10.3f%10.3f%10.3f%5d%s%10.3f\n",
                0.0, 0.0, obs_zenith, 0.0, 0.0, 0.0, 0, "     ", 0.0);
        // Card 3A1: IPARM=1, IPH=0, IDAY=80 (spring equinox), ISOURC=0
        fprintf(tp5, "%5d%5d%5d%5d\n", 1, 0, 80, 0);
        // Card 3A2: PARM1=lat(0), PARM2=lon(360), TIME=UTC_hour
        double utc_time = 12.0 + sun_zenith / 15.0;
        fprintf(tp5, "%10.3f%10.3f%10.3f%10.3f%10.3f%10.3f%10.3f%10.3f\n",
                0.0, 360.0, 0.0, 0.0, utc_time, 0.0, 0.0, 0.0);
    }

    // Write Card 3 for IEMSCT=3 (solar irradiance).
    // Simple format: sun zenith in ANGLE field, IDAY in day field. No IPARM cards.
    static void writeCard3_sun(FILE* tp5, double sun_zenith) {
        fprintf(tp5, "%10.3f%10.3f%10.3f%5d%s%10.3f%5d%10.3f\n",
                0.0, 0.0, sun_zenith, 80, "     ", 0.0, 0, 0.0);
    }

    void tp5sun(const std::filesystem::path& dir) {
        FILE* mr = fopen(wd(dir,"modroot.in").c_str(),"w"); fprintf(mr,"irrad"); fclose(mr);
        FILE* tp5 = fopen(wd(dir,"irrad.tp5").c_str(),"w");
        writeCard2(tp5, m_atmos, 3, 3, m_ihaze, getICLD(m_weather));
        writeCard3_sun(tp5, m_sun_zen_start);
        double w1,w2; getBands(w1,w2);
        fprintf(tp5,"%10.3f%10.3f%10.3f%10.3f%s%s%s\n",w1,w2,m_wl_step,m_wl_step*2.0,"RN","        ","W1     ");
        for (int i = 1; i < m_n_sun; i++) {
            fprintf(tp5, "%5d\n", 3);
            writeCard3_sun(tp5, m_sun_zen_start + i * m_sun_zen_step);
        }
        fprintf(tp5, "%5d\n", 0); fclose(tp5);
    }

    void tp5sky(const std::filesystem::path& dir) {
        FILE* mr = fopen(wd(dir,"modroot.in").c_str(),"w"); fprintf(mr,"skyRad"); fclose(mr);
        FILE* tp5 = fopen(wd(dir,"skyRad.tp5").c_str(),"w");
        writeCard2(tp5, m_atmos, 2, 3, m_ihaze, getICLD(m_weather));
        writeCard3_radiance(tp5, 0.0, m_sun_zen_start);
        double w1,w2; getBands(w1,w2);
        fprintf(tp5,"%10.3f%10.3f%10.3f%10.3f%s%s%s\n",w1,w2,m_wl_step,m_wl_step*2.0,"RN","        ","W1     ");
        for (int i = 1; i < m_n_sun; i++) {
            fprintf(tp5, "%5d\n", 3);
            writeCard3_radiance(tp5, 0.0, m_sun_zen_start + i * m_sun_zen_step);
        }
        fprintf(tp5, "%5d\n", 0); fclose(tp5);
    }

    void tp5trans(const std::filesystem::path& dir) {
        FILE* mr = fopen(wd(dir,"modroot.in").c_str(),"w"); fprintf(mr,"TRANS"); fclose(mr);
        FILE* tp5 = fopen(wd(dir,"TRANS.tp5").c_str(),"w");
        writeCard2(tp5, m_atmos, 2, 3, m_ihaze, getICLD(m_weather));
        writeCard3_radiance(tp5, 0.0, m_sun_zen_start);
        double w1,w2; getBands(w1,w2);
        fprintf(tp5,"%10.3f%10.3f%10.3f%10.3f%s%s%s\n",w1,w2,m_wl_step,m_wl_step*2.0,"TN","        ","W1     ");
        for (int i = 1; i < m_n_sun; i++) {
            fprintf(tp5, "%5d\n", 3);
            writeCard3_radiance(tp5, 0.0, m_sun_zen_start + i * m_sun_zen_step);
        }
        fprintf(tp5, "%5d\n", 0); fclose(tp5);
    }

    void saveSunData(const std::filesystem::path& dir, const char* outFile) {
        double wvStart=m_wl_start, wvEnd=m_wl_stop, wvgap=m_wl_step;
        int wn=(int)((wvEnd-wvStart)/wvgap)+1;
        std::vector<float> wvLen(wn); for(int i=0;i<wn;i++) wvLen[i]=(float)(wvStart+i*wvgap);
        char inFile[FILENAMELENGHTH];
        FILE* mr=fopen(wd(dir,"modroot.in").c_str(),"r"); fscanf(mr,"%s",inFile); fclose(mr);
        std::string plt=wd(dir,std::string(inFile)+".plt");
        std::vector<int> index(wn); GetIndex(index.data(),plt.c_str(),wvLen.data(),wn);
        // Sun data now has m_n_sun angle entries
        std::vector<MODATA> modData(m_n_sun*wn);
        GetModDataSolar(modData,plt.c_str(),index.data(),wn);
        FILE* pf=fopen(wd(dir,outFile).c_str(),"w");
        for(int i=0;i<m_n_sun;i++){
            fprintf(pf,"%d\n",i);
            for(int j=0;j<wn;j++) fprintf(pf,"%.1f\t%.4e\n",wvLen[j],modData[i*wn+j].data);
        }
        fclose(pf);
    }

    void saveSkyData(const std::filesystem::path& dir, const char* outFile) {
        double wvStart=m_wl_start, wvEnd=m_wl_stop, wvgap=m_wl_step;
        int n=(int)((wvEnd-wvStart)/wvgap)+1;
        std::vector<float> wvLen(n); for(int i=0;i<n;i++) wvLen[i]=(float)(wvStart+i*wvgap);
        char inFile[FILENAMELENGHTH];
        FILE* mr=fopen(wd(dir,"modroot.in").c_str(),"r"); fscanf(mr,"%s",inFile); fclose(mr);
        std::string plt=wd(dir,std::string(inFile)+".plt");
        std::vector<int> index(n); GetIndex(index.data(),plt.c_str(),wvLen.data(),n);
        std::vector<MODATA> modData(m_n_sun*n);
        GetModDataSolar(modData,plt.c_str(),index.data(),n);
        FILE* pf=fopen(wd(dir,outFile).c_str(),"w");
        for(int i=0;i<m_n_sun;i++){
            fprintf(pf,"%d\n",i);
            for(int j=0;j<n;j++) fprintf(pf,"%.1f\t%.4e\n",wvLen[j],modData[i*n+j].data);
        }
        fclose(pf);
    }

    void savetransData(const std::filesystem::path& dir, const char* outFile) {
        double wvStart=m_wl_start, wvEnd=m_wl_stop;
        int n=(int)((wvEnd-wvStart)/m_wl_step)+1;
        std::vector<float> wvLen(n); for(int i=0;i<n;i++) wvLen[i]=(float)(wvStart+i*m_wl_step);
        char inFile[FILENAMELENGHTH];
        FILE* mr=fopen(wd(dir,"modroot.in").c_str(),"r"); fscanf(mr,"%s",inFile); fclose(mr);
        std::string plt=wd(dir,std::string(inFile)+".plt");
        std::vector<int> index(n); GetIndex(index.data(),plt.c_str(),wvLen.data(),n);
        std::vector<MODATA> modData(m_n_sun*n);
        GetModDataSolar(modData,plt.c_str(),index.data(),n);
        FILE* pf=fopen(wd(dir,outFile).c_str(),"w");
        for(int i=0;i<m_n_sun;i++){
            fprintf(pf,"%d\n",i);
            for(int j=0;j<n;j++) fprintf(pf,"%.1f\t%.4e\n",wvLen[j],modData[i*n+j].data);
        }
        fclose(pf);
    }
};

} // namespace qltrans
