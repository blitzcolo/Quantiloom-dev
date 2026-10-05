#include "dataset/RigConfig.hpp"
#include "postprocess/CameraConfigIO.hpp"
#include "core/Sha256.hpp"
#include <algorithm>
#include <cmath>
#include <set>
#include <sstream>
#include <iomanip>
#include <glm/glm.hpp>

namespace quantiloom::dataset {
namespace {
bool Identifier(const String& s) {
    if(s.empty() || s.size()>128 || s=="." || s=="..") return false;
    return std::all_of(s.begin(),s.end(),[](unsigned char c) {
        return (c>='a' && c<='z') || (c>='A' && c<='Z') || (c>='0' && c<='9') || c=='_' || c=='-';
    });
}
Result<std::array<f64,16>,String> Pose(const Config& c,const char* key) {
    std::array<f64,16> p{1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};
    if(c.Has(key)) {
        const auto v=c.GetDoubleArray(key);
        if(v.size()!=16) return Result<std::array<f64,16>,String>::Err("pose requires 16 row-major values");
        std::copy(v.begin(),v.end(),p.begin());
    }
    for(auto x:p) if(!std::isfinite(x)) return Result<std::array<f64,16>,String>::Err("non-finite rig pose");
    if(p[12]!=0 || p[13]!=0 || p[14]!=0 || p[15]!=1)
        return Result<std::array<f64,16>,String>::Err("rig pose must be affine");
    const glm::dvec3 x(p[0],p[4],p[8]),y(p[1],p[5],p[9]),z(p[2],p[6],p[10]);
    if(std::abs(glm::dot(x,x)-1)>1e-8 || std::abs(glm::dot(y,y)-1)>1e-8 ||
       std::abs(glm::dot(z,z)-1)>1e-8 || std::abs(glm::dot(x,y))>1e-8 ||
       std::abs(glm::dot(x,z))>1e-8 || std::abs(glm::dot(y,z))>1e-8 ||
       glm::dot(glm::cross(x,y),z)<1-1e-8)
        return Result<std::array<f64,16>,String>::Err("rig pose must be a proper rigid transform");
    return p;
}
}

Result<RigConfig,String> ParseRigConfig(const Config& doc,const String& base) {
    const auto error=[](String s){return Result<RigConfig,String>::Err(std::move(s));};
    RigConfig rig;
    rig.version=doc.GetUInt("rig.version",0);
    rig.seed=doc.GetUInt("fusion.seed",0);
    rig.id=doc.GetString("rig.id"); rig.referenceCamera=doc.GetString("rig.reference_camera");
    if(rig.version!=1 || !Identifier(rig.id) || !Identifier(rig.referenceCamera))
        return error("rig requires version=1, a safe id and reference_camera");
    auto pose=Pose(doc,"rig.rig_to_world");
    if(!pose) return error(pose.error());
    rig.rigToWorld=*pose;
    std::set<String> ids;
    for(const auto& entry:doc.GetTableArray("rig.cameras")) {
        RigCamera camera;
        camera.id=entry.GetString("id");
        if(!Identifier(camera.id) || !ids.insert(camera.id).second) return error("invalid or duplicate camera id");
        auto transform=Pose(entry,"camera_to_rig");
        if(!transform) return error(transform.error());
        camera.cameraToRig=*transform;
        auto sensor=ParseCameraConfig(entry,SpectralMode::Single,base);
        if(!sensor) return error("camera "+camera.id+": "+sensor.error());
        if(!sensor.value().enabled || sensor.value().inputKind!=camera::CameraInputKind::SpectralMeasurement)
            return error("rig cameras require enabled spectral sensor configurations");
        if(!sensor.value().motion.keys.empty()) return error("fusion rig v1 requires static camera poses");
        camera.sensor=std::move(sensor.value());
        core::Sha256 seed;
        seed.UpdateU32(rig.seed);
        seed.UpdateString(rig.id); seed.UpdateString(camera.id);
        const auto digest=seed.FinalizeHex();
        camera.renderSeed=static_cast<u32>(std::stoul(digest.substr(0,8),nullptr,16));
        camera.sensor.randomSeed=camera.renderSeed;
        rig.cameras.push_back(std::move(camera));
    }
    if(rig.cameras.empty() || !ids.contains(rig.referenceCamera)) return error("rig has no cameras or unknown reference camera");
    std::set<std::pair<String,String>> pairs;
    if(doc.Has("rig.pairs")) {
        for(const auto& entry:doc.GetTableArray("rig.pairs")) {
            RigPair p{entry.GetString("source"),entry.GetString("target")};
            if(!ids.contains(p.sourceCamera) || !ids.contains(p.targetCamera) ||
                p.sourceCamera==p.targetCamera || !pairs.emplace(p.sourceCamera,p.targetCamera).second)
                return error("invalid or duplicate rig pair");
            rig.pairs.push_back(std::move(p));
        }
    } else for(const auto& camera:rig.cameras) if(camera.id!=rig.referenceCamera)
        rig.pairs.push_back({camera.id,rig.referenceCamera});
    return rig;
}

Result<Config,String> RigCameraScene(const Config& scene,const RigConfig& rig,const String& id) {
    const auto it=std::find_if(rig.cameras.begin(),rig.cameras.end(),[&](const RigCamera& c){return c.id==id;});
    if(it==rig.cameras.end()) return Result<Config,String>::Err("unknown rig camera");
    std::array<f64,16> pose{};
    for(size_t r=0;r<4;++r) for(size_t c=0;c<4;++c) for(size_t k=0;k<4;++k)
        pose[r*4+c]+=rig.rigToWorld[r*4+k]*it->cameraToRig[k*4+c];
    std::ostringstream text; text<<std::setprecision(17);
    text<<"[camera]\nposition = ["<<pose[3]<<","<<pose[7]<<","<<pose[11]<<"]\n";
    text<<"look_at = ["<<pose[3]+pose[2]<<","<<pose[7]+pose[6]<<","<<pose[11]+pose[10]<<"]\n";
    text<<"up = ["<<-pose[1]<<","<<-pose[5]<<","<<-pose[9]<<"]\nprojection = \"perspective\"\n";
    text<<"[renderer]\nseed = "<<it->renderSeed<<"\nresolution = ["
        <<it->sensor.optics.sensorWidthPx<<","<<it->sensor.optics.sensorHeightPx<<"]\n";
    text<<"[dataset]\nmetadata = true\n"<<CameraConfigToToml(it->sensor);
    const auto overrides=Config::Parse(text.str());
    if(!overrides) return Result<Config,String>::Err(overrides.error());
    return scene.MergedWith(*overrides);
}

String RigConfigToToml(const RigConfig& rig) {
    std::ostringstream out;out<<std::setprecision(17);
    const auto matrix=[&](const std::array<f64,16>& values) {
        out<<"[";
        for(size_t i=0;i<values.size();++i) {if(i) out<<", ";out<<values[i];}
        out<<"]\n";
    };
    out<<"[fusion]\nseed = "<<rig.seed<<"\n[rig]\nversion = "<<rig.version
        <<"\nid = "<<std::quoted(rig.id)<<"\nreference_camera = "<<std::quoted(rig.referenceCamera)<<"\nrig_to_world = ";
    matrix(rig.rigToWorld);
    if(rig.pairs.empty())out<<"pairs = []\n";
    for(const auto& c:rig.cameras) {
        out<<"\n[[rig.cameras]]\nid = "<<std::quoted(c.id)<<"\ncamera_to_rig = ";matrix(c.cameraToRig);
        auto frozen=c.sensor;
        frozen.optics.knownPsfSourcePath.clear();
        frozen.photon.nucGainMapPath.clear();frozen.photon.nucOffsetElectronsMapPath.clear();
        frozen.thermal.nucGainMapPath.clear();frozen.thermal.nucOffsetDnMapPath.clear();
        frozen.isp.defectPixelsPath.clear();
        for(auto& channel:frozen.device.channels)for(auto* curve:{&channel.response.lensTransmission,
            &channel.response.filterTransmission,&channel.response.quantumEfficiency,
            &channel.response.thermalAbsorptance,&channel.response.systemResponse})
            if(*curve)(*curve)->dataPath.clear();
        std::istringstream sensor(CameraConfigToToml(frozen));String line;
        while(std::getline(sensor,line)) {
            if(!line.empty() && line.front()=='[') {
                const size_t position=line.find_first_not_of('[');
                line.insert(position,"rig.cameras.");
            }
            out<<line<<'\n';
        }
    }
    for(const auto& p:rig.pairs) out<<"\n[[rig.pairs]]\nsource = "<<std::quoted(p.sourceCamera)
        <<"\ntarget = "<<std::quoted(p.targetCamera)<<"\n";
    return out.str();
}
} // namespace quantiloom::dataset
