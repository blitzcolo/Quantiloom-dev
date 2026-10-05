#include "renderer/FusionTransport.hpp"
#include <map>
#include <array>
#include <cmath>
#include <set>
#include <algorithm>

namespace quantiloom::rendercore {
namespace {
struct Surface {FusionTransportGpu gpu;};
Result<f32,String> ClosedOrientation(const Mesh& mesh) {
    std::map<std::array<f32,3>,u32> vertices;
    std::map<std::pair<u32,u32>,std::pair<u32,i32>> edges;
    f64 volume=0;
    for(const auto& primitive:mesh.primitives) {
      if(primitive.indices.size()%3)return Result<f32,String>::Err("solid index count must be divisible by three");
      for(size_t i=0;i<primitive.indices.size();i+=3) {
        glm::dvec3 p[3];u32 ids[3];
        for(u32 j=0;j<3;++j) {
            if(primitive.indices[i+j]>=primitive.positions.size()) return Result<f32,String>::Err("invalid solid triangle");
            const auto v=primitive.positions[primitive.indices[i+j]];
            if(!std::isfinite(v.x)||!std::isfinite(v.y)||!std::isfinite(v.z)) return Result<f32,String>::Err("non-finite solid vertex");
            const std::array<f32,3> key{v.x,v.y,v.z};
            if(!vertices.contains(key)) vertices.emplace(key,static_cast<u32>(vertices.size()));
            ids[j]=vertices.at(key);p[j]=glm::dvec3(v);
        }
        if(glm::length(glm::cross(p[1]-p[0],p[2]-p[0]))<1e-15) return Result<f32,String>::Err("degenerate solid triangle");
        volume+=glm::dot(p[0],glm::cross(p[1],p[2]))/6;
        for(u32 j=0;j<3;++j) {
            const u32 a=ids[j],b=ids[(j+1)%3];
            auto& e=edges[{std::min(a,b),std::max(a,b)}];++e.first;e.second+=a<b ? 1 : -1;
        }
      }
    }
    if(edges.empty() || !std::isfinite(volume) || std::abs(volume)<1e-15)
        return Result<f32,String>::Err("solid has no enclosed volume");
    for(const auto& [key,edge]:edges) if(edge.first!=2 || edge.second!=0)
        return Result<f32,String>::Err("solid mesh must be closed, manifold and consistently wound");
    return volume>0 ? 1.0f : -1.0f;
}
}

Result<Vector<u32>,String> InitialFusionMedia(const Scene& scene,
    const Vector<FusionTransportGpu>& records,const glm::vec3& origin) {
    std::map<u32,u32> solidNodes;
    for(u32 i=1;i<records.size();++i) if(records[i].mode==2) solidNodes.emplace(records[i].nodeId-1,i);
    std::vector<std::pair<f64,u32>> inside;
    const glm::dvec3 worldDirection=glm::normalize(glm::dvec3(.312731,.717127,1.0));
    for(const auto& [nodeIndex,record]:solidNodes) {
        const auto& node=scene.nodes[nodeIndex];
        const auto inverse=glm::inverse(glm::dmat4(node.transform));
        const glm::dvec3 o=inverse*glm::dvec4(origin,1),d=inverse*glm::dvec4(worldDirection,0);
        std::vector<f64> crossings;
        for(const auto& primitive:scene.meshes[node.meshIndex].primitives)
            for(size_t i=0;i<primitive.indices.size();i+=3) {
                const glm::dvec3 a(primitive.positions[primitive.indices[i]]),
                    b(primitive.positions[primitive.indices[i+1]]),c(primitive.positions[primitive.indices[i+2]]);
                const auto e1=b-a,e2=c-a,h=glm::cross(d,e2);
                const f64 det=glm::dot(e1,h);if(std::abs(det)<1e-14)continue;
                const auto offset=o-a;const f64 u=glm::dot(offset,h)/det;
                if(u<0 || u>1)continue;
                const auto q=glm::cross(offset,e1);const f64 v=glm::dot(d,q)/det;
                if(v<0 || u+v>1)continue;
                const f64 t=glm::dot(e2,q)/det;
                if(std::abs(t)<1e-8) return Result<Vector<u32>,String>::Err("camera lies on a fusion medium boundary");
                if(t>0)crossings.push_back(t);
            }
        std::sort(crossings.begin(),crossings.end());
        crossings.erase(std::unique(crossings.begin(),crossings.end(),[](f64 a,f64 b){return std::abs(a-b)<1e-9*std::max(1.0,std::abs(a));}),crossings.end());
        if(crossings.size()%2) inside.emplace_back(crossings.front(),record);
    }
    if(inside.size()>8)return Result<Vector<u32>,String>::Err("camera begins inside more than eight nested media");
    std::sort(inside.rbegin(),inside.rend());
    Vector<u32> result;for(const auto& [distance,record]:inside)result.push_back(record);
    return result;
}

Result<Vector<FusionTransportGpu>,String> ResolveFusionTransport(const Config& config,Scene& scene) {
    Vector<FusionTransportGpu> styles(scene.materials.size());
    auto overrides=config.GetTable("material_overrides");
    bool any=false;
    for(size_t i=0;i<scene.materials.size();++i) {
        auto& material=scene.materials[i];
        Config entry;
        for(const auto& table:config.GetTableArray("materials"))
            if(table.GetString("name")==material.name) entry=entry.MergedWith(table);
        if(overrides) {auto table=overrides.value().GetNamedTable(material.name);if(table) entry=entry.MergedWith(*table);}
        const String mode=entry.GetString("fusion_transport","legacy");
        auto& style=styles[i];style.materialId=static_cast<u32>(i);
        if(mode=="legacy") continue;
        if(mode!="thin_sheet" && mode!="solid") return Result<Vector<FusionTransportGpu>,String>::Err("unsupported fusion transport model for "+material.name);
        any=true;style.mode=mode=="solid" ? 2u : 1u;
        style.absorptionPerMeter=entry.GetFloat("fusion_absorption_m_inv",0);
        style.sheetReflectance=entry.GetFloat("fusion_sheet_reflectance",0);
        style.sheetTransmittance=entry.GetFloat("fusion_sheet_transmittance",0);
        if(entry.Has("fusion_sheet_reflectance")) style.flags|=2;
        if(entry.Has("fusion_sheet_transmittance")) style.flags|=4;
        if(!std::isfinite(style.absorptionPerMeter)||style.absorptionPerMeter<0 ||
           !std::isfinite(style.sheetReflectance)||!std::isfinite(style.sheetTransmittance)||
           style.sheetReflectance<0||style.sheetTransmittance<0||style.sheetReflectance+style.sheetTransmittance>1)
            return Result<Vector<FusionTransportGpu>,String>::Err("invalid fusion transport coefficients for "+material.name);
        if(material.alphaMode!=Material::AlphaMode::Opaque || material.scatteringCoeff>0 ||
           material.volumeDensity>0 || material.roughnessFactor>1e-5)
            return Result<Vector<FusionTransportGpu>,String>::Err("fusion transport requires smooth opaque-coverage non-scattering interfaces");
        if(style.mode==2 && (material.temperatureTextureIndex>=0 ||
            !material.temperatureTexturePath.empty() || config.GetBool("thermal.enabled",false)))
            return Result<Vector<FusionTransportGpu>,String>::Err("solid fusion medium requires a uniform authored temperature");
        if(material.fluorescenceYield>0 || (style.mode==2 &&
            (!material.emissiveCurveSource.empty() || glm::length(material.emissiveFactor)>0 || material.metallicFactor>1e-5)))
            return Result<Vector<FusionTransportGpu>,String>::Err("solid media cannot carry surface emission/metal lobes; fluorescent interfaces are unsupported");
        if(style.mode==2 && !entry.Has("fusion_absorption_m_inv")) style.flags|=1;
        if(!std::isfinite(material.ior) || material.ior<=0 || !std::isfinite(material.irTemperature_K) || material.irTemperature_K<0)
            return Result<Vector<FusionTransportGpu>,String>::Err("fusion medium requires a positive finite index and nonnegative temperature");
        if(style.mode==1 && (((style.flags&2)==0 && material.spectralReflectanceCurveIndex<0 &&
            material.irReflectanceCurve.empty() && !material.HasQuantiloomRef()) ||
            ((style.flags&4)==0 && material.irTransmittanceCurve.empty())))
            return Result<Vector<FusionTransportGpu>,String>::Err("thin sheet requires explicit reflectance and transmittance or covering spectral data");
        material.doubleSided=true;
    }
    Vector<FusionTransportGpu> records(1);
    records[0].mode=any ? 1 : 0;
    for(size_t n=0;n<scene.nodes.size();++n) {
        const auto& node=scene.nodes[n];if(!node.active) continue;
        if(node.meshIndex>=scene.meshes.size()) {
            if(any)return Result<Vector<FusionTransportGpu>,String>::Err("invalid fusion node mesh");
            continue;
        }
        const auto& mesh=scene.meshes[node.meshIndex];
        bool solid=false;std::set<u32> materials;
        for(const auto& p:mesh.primitives) {
            if(p.materialId>=styles.size()) {
                if(any)return Result<Vector<FusionTransportGpu>,String>::Err("invalid fusion material index");
                continue;
            }
            materials.insert(p.materialId);solid|=styles[p.materialId].mode==2;
        }
        f32 orientation=1;
        if(solid) {
            if(materials.size()!=1) return Result<Vector<FusionTransportGpu>,String>::Err("solid node must have one homogeneous material");
            const auto closed=ClosedOrientation(mesh);
            if(!closed) return Result<Vector<FusionTransportGpu>,String>::Err(node.name+": "+closed.error());
            orientation=*closed;
        }
        for(const auto& p:mesh.primitives) {
            auto record=p.materialId<styles.size() ? styles[p.materialId] : FusionTransportGpu{};
            record.nodeId=static_cast<u32>(n+1);record.orientation=orientation;
            records.push_back(record);
        }
    }
    return records;
}
}
