#include "dataset/OpticalCorrespondence.hpp"
#include <nlohmann/json.hpp>
#include <cstring>
#include <map>
#include <set>
#include <algorithm>
#include <limits>

namespace quantiloom::dataset {
Vector<OpticalEndpoint> DecodeOpticalEndpoints(const FusionPathChunk& chunk,
    const String& productId,const ProductGeometry& geometry) {
    Vector<OpticalEndpoint> output;
    const auto u=[&](size_t offset){u32 v;std::memcpy(&v,chunk.bytes.data()+offset,4);return v;};
    const auto f=[&](size_t offset){f32 v;std::memcpy(&v,chunk.bytes.data()+offset,4);return v;};
    if(chunk.bytes.size()!=128+static_cast<size_t>(chunk.storedRays)*(32+chunk.slotsPerRay*80)) return output;
    for(u32 i=0;i<chunk.storedRays;++i) {
        const size_t ray=128+i*32;const u32 end=u(ray+28);
        if((end&0x80000000u)==0 || u(ray+24)!=0)continue;
        const u32 depth=end&0x7FFFFFFFu;if(depth>=chunk.slotsPerRay)continue;
        const size_t vertex=128+static_cast<size_t>(chunk.storedRays)*32+(i*chunk.slotsPerRay+depth)*80;
        if(u(vertex+32)!=3)continue;
        OpticalEndpoint e;e.productId=productId;e.row=i;e.pixel=u(ray+16);
        e.nodeId=u(vertex+36);e.primitiveId=u(vertex+40);e.wavelengthNm=chunk.wavelengthNm;
        e.acquisitionIndex=chunk.acquisitionIndex;
        e.position={f(vertex),f(vertex+4),f(vertex+8)};
        const size_t first=128+static_cast<size_t>(chunk.storedRays)*32+i*chunk.slotsPerRay*80;
        const glm::dvec3 hit(f(first),f(first+4),f(first+8));
        const auto offset=hit-glm::dvec3(geometry.camera.origin);
        const auto projected=camera::ProjectDirection(*geometry.nativeProjection,
            {glm::dot(offset,glm::dvec3(geometry.camera.right)),
             -glm::dot(offset,glm::dvec3(geometry.camera.up)),glm::dot(offset,glm::dvec3(geometry.camera.forward))});
        if(!projected.valid)continue;e.nativePixel=projected.pixel;e.throughOptics=depth>0;
        for(u32 d=0;d<depth;++d) {
            const size_t address=first+d*80;
            const glm::dvec3 normal(f(address+16),f(address+20),f(address+24));
            const glm::dvec3 outgoing(f(address+48),f(address+52),f(address+56));
            if(glm::dot(normal,outgoing)>0)e.branchMask|=1u<<d;
        }
        output.push_back(std::move(e));
    }
    return output;
}

Result<String,String> MatchOpticalPaths(OfflineRenderer& target,
    const Vector<OpticalEndpoint>& source,const Vector<OpticalEndpoint>& samples,
    const ProductGeometry& geometry,u32 limit,const std::function<bool()>& cancelled) {
    using Json=nlohmann::json;
    struct Trial {size_t source;f64 lambda;u32 mask;glm::dvec2 pixel;bool active=true;};
    const bool optical=std::any_of(source.begin(),source.end(),[](const auto& e){return e.throughOptics;}) ||
        std::any_of(samples.begin(),samples.end(),[](const auto& e){return e.throughOptics;});
    Json out={{"schema","quantiloom.fusion.path_correspondence"},{"schema_version",2},
        {"complete_solution_set",false},{"unmatched_is_unresolved",true},
        {"meaning","geometric_path_pair_not_detectability"},{"rows",Json::array()},
        {"source_recorded_paths",source.size()},{"source_limit",limit},
        {"solver","bounded_multiseed_forward_ray_newton"},{"residual_unit","m"}};
    if(!optical || source.empty() || samples.empty() || !limit) return out.dump();
    std::set<f64> wavelengths;for(const auto& s:samples)wavelengths.insert(s.wavelengthNm);
    Vector<size_t> selected;
    const size_t stride=std::max<size_t>(1,(source.size()+limit-1)/limit);
    for(size_t i=0;i<source.size();i+=stride)selected.push_back(i);
    Vector<Trial> trials;
    std::map<size_t,size_t> rowForSource;
    const auto& c=geometry.camera;
    for(auto i:selected) {
        rowForSource[i]=out["rows"].size();
        out["rows"].push_back({{"source_path_product",source[i].productId},{"source_ray_row",source[i].row},
             {"source_pixel",source[i].pixel},{"source_wavelength_nm",source[i].wavelengthNm},
             {"source_native_pixel",{source[i].nativePixel.x,source[i].nativePixel.y}},
             {"source_branch_mask",source[i].branchMask},{"unique_solution_proven",false},
            {"node_id",source[i].nodeId},{"primitive_id",source[i].primitiveId},
            {"status","unresolved"},{"matches",Json::array()}});
        for(auto lambda:wavelengths) {
            const auto delta=source[i].position-glm::dvec3(c.origin);
            const auto p=camera::ProjectDirection(*geometry.nativeProjection,
                {glm::dot(delta,glm::dvec3(c.right)),-glm::dot(delta,glm::dvec3(c.up)),glm::dot(delta,glm::dvec3(c.forward))});
            if(p.valid)trials.push_back({i,lambda,0,p.pixel});
            Vector<std::pair<f64,const OpticalEndpoint*>> near;
            for(const auto& s:samples) if(s.wavelengthNm==lambda && s.nodeId==source[i].nodeId && s.primitiveId==source[i].primitiveId)
                near.push_back({glm::dot(s.position-source[i].position,s.position-source[i].position),&s});
            std::sort(near.begin(),near.end(),[](const auto& a,const auto& b){return a.first<b.first;});
            std::set<u32> masks;
            for(const auto& [distance,endpoint]:near)if(masks.insert(endpoint->branchMask).second) {
                trials.push_back({i,lambda,endpoint->branchMask,endpoint->nativePixel});
                if(masks.size()==3)break;
            }
        }
    }
    for(u32 iteration=0;iteration<12;++iteration) {
        if(cancelled && cancelled())return Result<String,String>::Err("fusion export cancelled");
        Vector<OpticalProbe> probes;Vector<size_t> active;
        constexpr f64 step=.05;
        for(size_t i=0;i<trials.size();++i)if(trials[i].active) {
            const auto& t=trials[i];active.push_back(i);
            for(auto offset:{glm::dvec2(0),glm::dvec2(step,0),glm::dvec2(0,step)})
                probes.push_back({glm::vec2(t.pixel+offset),t.lambda,t.mask});
        }
        if(probes.empty())break;
        auto result=target.QueryOpticalPaths(probes,geometry.referenceTimeSeconds);
        if(!result)return Result<String,String>::Err(result.error());
        for(size_t n=0;n<active.size();++n) {
            auto& t=trials[active[n]];const auto& goal=source[t.source];
            const auto& value=result.value()[n*3];
            if(value.flags || value.surface.instanceId!=goal.nodeId || value.primitiveId!=goal.primitiveId || value.throughput<=1e-12) {t.active=false;continue;}
            const glm::dvec3 point(value.surface.worldPosition),error=goal.position-point;
            const f64 residual=glm::length(error)*geometry.worldUnitsToMeters;
            const f64 magnitude=std::max({1.0,std::abs(goal.position.x),std::abs(goal.position.y),std::abs(goal.position.z)});
            const f64 tolerance=std::max(1e-4,8*std::numeric_limits<f32>::epsilon()*magnitude*geometry.worldUnitsToMeters);
            if(residual<=tolerance) {
                auto& row=out["rows"][rowForSource.at(t.source)];bool duplicate=false;
                for(const auto& m:row["matches"]) {
                    const glm::dvec2 old(m["target_pixel"][0].get<f64>(),m["target_pixel"][1].get<f64>());
                    if(m["target_wavelength_nm"]==t.lambda && m["target_branch_mask"]==t.mask && glm::length(old-t.pixel)<.01)duplicate=true;
                }
                if(!duplicate)row["matches"].push_back({{"target_pixel",{t.pixel.x,t.pixel.y}},
                    {"target_wavelength_nm",t.lambda},{"target_branch_mask",t.mask},
                    {"forward_residual_m",residual},{"throughput",value.throughput}});
                row["status"]="verified_candidates";t.active=false;continue;
            }
            const auto& dx=result.value()[n*3+1];const auto& dy=result.value()[n*3+2];
            if(dx.flags || dy.flags || dx.surface.instanceId!=goal.nodeId || dy.surface.instanceId!=goal.nodeId ||
                dx.primitiveId!=goal.primitiveId || dy.primitiveId!=goal.primitiveId) {t.active=false;continue;}
            const glm::dvec3 jx=(glm::dvec3(dx.surface.worldPosition)-point)/step,
                jy=(glm::dvec3(dy.surface.worldPosition)-point)/step;
            const f64 a=glm::dot(jx,jx),b=glm::dot(jx,jy),d=glm::dot(jy,jy),det=a*d-b*b;
            if(!std::isfinite(det)||det<=1e-20){t.active=false;continue;}
            const f64 ex=glm::dot(jx,error),ey=glm::dot(jy,error);
            glm::dvec2 update((d*ex-b*ey)/det,(a*ey-b*ex)/det);
            const f64 length=glm::length(update);if(length>8)update*=8/length;
            t.pixel+=update;
            if(t.pixel.x<0 || t.pixel.y<0 || t.pixel.x>=geometry.width || t.pixel.y>=geometry.height)t.active=false;
        }
    }
    return out.dump();
}
}
