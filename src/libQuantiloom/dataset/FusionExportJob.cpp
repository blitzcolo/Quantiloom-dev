#include "dataset/FusionExportJob.hpp"
#include "dataset/ExportSession.hpp"
#include "dataset/OpticalCorrespondence.hpp"
#include "dataset/RectificationSupport.hpp"
#include "io/ImageIO.hpp"
#include "postprocess/CameraConfigIO.hpp"
#include "postprocess/CameraPhysics.hpp"
#include "core/Sha256.hpp"
#include "renderer/RenderDevice.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <set>
#include <map>
#include <cmath>
#include <tuple>
#include <limits>
#include <algorithm>

namespace quantiloom::dataset {
namespace {
using Json=nlohmann::json;
namespace fs=std::filesystem;
String SignalName(camera::SignalKind s) {
    using K=camera::SignalKind;
    switch(s) {
    case K::SpectralRadiance: return "spectral_radiance";
    case K::BandMeasurement: return "response_weighted_measurement";
    case K::DeviceLinear: return "device_linear";
    case K::CieLinearSrgb: return "cie_linear_srgb";
    case K::DisplaySrgb: return "display_srgb";
    case K::RawDN: return "raw_dn";
    case K::ApparentTemperature: return "apparent_temperature";
    case K::FastRgbApproximation: return "fast_rgb_approximation";
    }
    throw std::runtime_error("unsupported signal kind");
}
Json Signal(const camera::CameraProduct& p) {
    const auto& s=p.signal;
    return {{"kind",SignalName(s.kind)},{"unit",s.unit},
        {"channels",p.image.channelNames},{"storage","float32"},
        {"channel_wavelength_nm",s.channelWavelengthNm},
        {"response_span_nm",s.channelResponseSpanNm},{"response_ids",s.channelResponseIds},
        {"integration",s.kind==camera::SignalKind::BandMeasurement ? "declared_camera_response" :
            s.kind==camera::SignalKind::SpectralRadiance ? "per_wavelength" : "device_pipeline"},
        {"colour_space",s.kind==camera::SignalKind::CieLinearSrgb || s.kind==camera::SignalKind::DisplaySrgb ? "sRGB" : "device_native"},
        {"transfer",s.kind==camera::SignalKind::DisplaySrgb ? "sRGB" : "linear"},
        {"cfa",static_cast<u32>(s.cfa)},{"calibration_status",static_cast<u32>(s.calibration)}};
}
camera::CameraProjection RectifiedProjection(const camera::CameraProjection& native,u32 w,u32 h) {
    auto p=native;p.model=camera::ProjectionModel::Pinhole;p.coefficients={};
    p.explicitIntrinsics=true;p.cx=w*0.5;p.cy=h*0.5;
    return p;
}
Image Rectify(const Image& input,const camera::CameraProjection& native,
    const camera::CameraProjection& target,Image& mapping,Image& valid) {
    Image output(input.width,input.height,input.channels);
    output.channelNames=input.channelNames;output.metadata=input.metadata;
    mapping=Image(input.width,input.height,2);mapping.channelNames={"native_x","native_y"};
    valid=Image(input.width,input.height,1);valid.channelNames={"valid"};
    for(u32 y=0;y<input.height;++y) for(u32 x=0;x<input.width;++x) {
        const auto ray=camera::UnprojectPixel(target,{x+0.5,y+0.5});
        if(!ray.valid) continue;
        const auto q=camera::ProjectDirection(native,ray.direction);
        if(!q.valid || !detail::HasRectificationSupport(native,q.pixel,input.width,input.height)) continue;
        const double u=q.pixel.x-.5,v=q.pixel.y-.5;
        const u32 x0=static_cast<u32>(u),y0=static_cast<u32>(v);
        const u32 x1=std::min(x0+1,input.width-1),y1=std::min(y0+1,input.height-1);
        const double a=u-x0,b=v-y0;
        for(u32 c=0;c<input.channels;++c) output(x,y,c)=static_cast<f32>(
            (1-b)*((1-a)*input(x0,y0,c)+a*input(x1,y0,c))+
            b*((1-a)*input(x0,y1,c)+a*input(x1,y1,c)));
        mapping(x,y,0)=static_cast<f32>(q.pixel.x);mapping(x,y,1)=static_cast<f32>(q.pixel.y);
        valid(x,y,0)=1;
    }
    return output;
}
bool SafeId(const String& s) {
    return !s.empty() && s!="." && s!=".." && s.size()<=128 &&
        s.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_- ")==String::npos &&
        s.find(' ')==String::npos;
}
void WriteText(const String& path,const String& text) {
    std::ofstream out(fs::path(path),std::ios::binary);
    out<<text; out.flush();
    if(!out) throw std::runtime_error("cannot write fusion artifact");
    out.close();
    if(!out) throw std::runtime_error("cannot close fusion artifact");
}
void InlineResponses(camera::CameraConfig& c) {
    c.optics.knownPsfSourcePath.clear();
    c.photon.nucGainMapPath.clear(); c.photon.nucOffsetElectronsMapPath.clear();
    c.thermal.nucGainMapPath.clear(); c.thermal.nucOffsetDnMapPath.clear();
    c.isp.defectPixelsPath.clear();
    for(auto& channel:c.device.channels) {
        auto& r=channel.response;
        for(auto* p:{&r.lensTransmission,&r.filterTransmission,&r.quantumEfficiency,
                    &r.thermalAbsorptance,&r.systemResponse}) if(*p) (*p)->dataPath.clear();
    }
}
}

Result<FusionExportResult,String> FusionExportJob::Run(const Config& scene,
    const RigConfig& authored,const FusionExportOptions& options) {
    try {
        if(!SafeId(options.sampleId) || options.outputDirectory.empty() ||
           !std::isfinite(options.referenceTimeSeconds)) throw std::runtime_error("invalid fusion output identity or time");
        // Validate caller-created rigs through the same parser as file jobs.
        if(authored.version!=1 || authored.cameras.empty()) throw std::runtime_error("invalid fusion rig");
        if(scene.HasSection("timeline") || !scene.GetTableArray("models.motion.keys").empty())
            throw std::runtime_error("fusion export v1 requires a static scene snapshot");
        RigConfig rig=authored;
        std::set<String> cameraIds;
        for(auto& c:rig.cameras) {
            if(!SafeId(c.id) || !cameraIds.insert(c.id).second) throw std::runtime_error("invalid or duplicate camera identity");
            InlineResponses(c.sensor);
            // A native sensor observation anchors truth; auxiliary observer
            // renders retain separate acquisitions even when other products are off.
            c.sensor.products.bandMeasurement=true;
            const auto valid=camera::ValidateCameraConfig(c.sensor);
            if(!valid) throw std::runtime_error(valid.error());
        }
        const auto canonicalDocument=Config::Parse(RigConfigToToml(rig));
        if(!canonicalDocument) throw std::runtime_error(canonicalDocument.error());
        const auto canonicalRig=ParseRigConfig(*canonicalDocument);
        if(!canonicalRig) throw std::runtime_error(canonicalRig.error());
        rig=*canonicalRig;
        if(!cameraIds.contains(rig.referenceCamera)) throw std::runtime_error("unknown fusion reference camera");
        Json manifest={{"schema","quantiloom.fusion.sample"},{"schema_version",2},
            {"sample_id",options.sampleId},{"rig_id",rig.id},{"reference_camera",rig.referenceCamera},
            {"reference_time_s",options.referenceTimeSeconds},{"observations",Json::array()},
            {"ground_truth",Json::array()},{"pairs",Json::array()},
            {"inference_requires_renderer",false},{"replayable",false}};
        const auto sceneText=scene.ToToml();
        manifest["scene_id"]=core::Sha256Hex(sceneText.data(),sceneText.size());
        manifest["scene_id_definition"]="frozen_configuration_sha256";
        if(options.dryRun) {
            for(const auto& c:rig.cameras) {
                auto config=RigCameraScene(scene,rig,c.id);
                if(!config) throw std::runtime_error(config.error());
                manifest["observations"].push_back({{"camera_id",c.id},{"seed",c.renderSeed},
                    {"width",c.sensor.optics.sensorWidthPx},{"height",c.sensor.optics.sensorHeightPx}});
            }
            return FusionExportResult{"","",manifest.dump(2)};
        }
        const fs::path directory=fs::absolute(fs::path(options.outputDirectory));
        const String manifestName=options.sampleId+".manifest.json";
        auto transaction=ExportSession::Create((directory/manifestName).string(),scene,
            {Json{{"export_schema_version",2},{"rig_id",rig.id},{"sample_id",options.sampleId},
                  {"reference_time_s",options.referenceTimeSeconds}}.dump()});
        if(!transaction) throw std::runtime_error(transaction.error());
        auto& session=*transaction.value();
        auto init=options.renderer;
        std::unique_ptr<RenderDevice> ownedDevice;
        if(!init.sharedDevice) {
            RenderDevice::InitParams deviceParams;deviceParams.pipelineCachePath=init.pipelineCachePath;
            auto created=RenderDevice::Create(deviceParams);
            if(!created) throw std::runtime_error(created.error());
            ownedDevice=std::move(*created);init.sharedDevice=ownedDevice.get();
        }
        manifest["record_id"]=session.RecordId();
        manifest["record_path"]=session.SidecarName();
        std::map<String,String> primaryProducts;
        std::map<String,GeometryTruth> geometries;
        std::map<String,Vector<OpticalEndpoint>> endpoints;
        u32 completed=0;
        for(const auto& c:rig.cameras) {
            if(options.cancelled && options.cancelled()) throw std::runtime_error("fusion export cancelled");
            if(options.onProgress) options.onProgress({completed,static_cast<u32>(rig.cameras.size()),c.id,"capture"});
            auto config=RigCameraScene(scene,rig,c.id);
            if(!config) throw std::runtime_error(config.error());
            auto cameraInit=init;
            cameraInit.fusionMaxRecordedRays=options.maxRecordedRays;
            Json pathChunks=Json::array();
            ProductGeometry pathGeometry;
            u32 chunkIndex=0;
            cameraInit.onFusionPathChunk=[&](const FusionPathChunk& chunk) {
                if(options.cancelled && options.cancelled()) throw std::runtime_error("fusion export cancelled");
                const String id=options.sampleId+"/"+c.id+"/paths/"+std::to_string(chunkIndex);
                ++chunkIndex;
                auto decoded=DecodeOpticalEndpoints(chunk,id,pathGeometry);
                auto& cache=endpoints[c.id];cache.insert(cache.end(),decoded.begin(),decoded.end());
            };
            auto renderer=OfflineRenderer::Create(*config,cameraInit);
            if(!renderer) throw std::runtime_error(renderer.error());
            auto truth=renderer.value()->CaptureGeometry(options.referenceTimeSeconds);
            if(!truth) throw std::runtime_error(truth.error());
            pathGeometry=truth.value().geometry;
            auto moved=renderer.value()->SetTimelineTime(options.referenceTimeSeconds);
            if(!moved) throw std::runtime_error(moved.error());
            camera::CaptureState state;
            if(c.sensor.warmup.seconds>0) {
                auto warm=renderer.value()->WarmUpCamera(state,c.sensor.warmup.seconds,c.sensor.readout.framePeriodSeconds);
                if(!warm) throw std::runtime_error(warm.error());
            }
            chunkIndex=0;endpoints[c.id].clear();pathChunks=Json::array();
            FusionCaptureOptionsV2 captureOptions;captureOptions.maxRecordedRays=options.maxRecordedRays;
            captureOptions.cancelled=options.cancelled;
            auto captured=renderer.value()->CaptureFusionV2(state,options.referenceTimeSeconds,captureOptions);
            if(!captured) throw std::runtime_error(captured.error());
            auto& images=captured.value().products;
            const auto geometryJson=truth.value().geometry.ToJson();
            if(!geometryJson) throw std::runtime_error(geometryJson.error());
            const Json geometry=Json::parse(*geometryJson);
            const std::tuple<Image*,const char*,const char*> geometryImages[]={
                {&truth.value().rayDistanceMeters,"ray_distance","m"},
                {&truth.value().cameraDepthMeters,"camera_z","m"},
                {&truth.value().worldPosition,"world_position","world_units"},
                {&truth.value().worldNormal,"geometric_normal","dimensionless"},
                {&truth.value().validity,"geometry_validity","enum"}};
            Json truthRecord={{"camera_id",c.id},{"products",Json::array()},
                {"path_chunks",pathChunks},
                {"instances",Json::parse(truth.value().instancesJson)},
                {"geometry",geometry},{"sampling","instantaneous_pixel_centre"},
                {"validity_codes",{{"miss",0},{"opaque",1},{"invalid_lens",2},{"partial_coverage",3},{"transmissive",4}}}};
            // Replace the callback's v1 artifacts with the versioned column records.
            // The callback is used only for matching sampled optical endpoints.
            truthRecord["path_chunks"]=Json::array();
            for(size_t i=0;i<captured.value().paths.size();++i) {
                const auto& chunk=captured.value().paths[i];
                const String id=options.sampleId+"/"+c.id+"/paths/"+std::to_string(i);
                const String path="ground_truth/"+c.id+"/paths_"+std::to_string(i)+".bin";
                auto staged=session.StagingPath(path);if(!staged)throw std::runtime_error(staged.error());
                std::ofstream stream(fs::path(*staged),std::ios::binary);
                stream.write(reinterpret_cast<const char*>(chunk.bytes.data()),chunk.bytes.size());stream.close();
                if(!stream)throw std::runtime_error("cannot write fusion column records");
                const auto description=Json::parse(chunk.descriptionJson);
                auto registered=session.RegisterFile(path,id,description.dump());
                if(!registered)throw std::runtime_error(registered.error());
                truthRecord["path_chunks"].push_back({{"product_id",id},{"path",path},{"description",description}});
            }
            const String measurementUnit=c.sensor.device.detector==camera::DetectorKind::Photon ? "e-/s" : "W";
            truthRecord["contribution_products"]=Json::array();
            for(u32 component=0;component<6;++component) {
                const char* names[]={"contribution_direct","contribution_reflected","contribution_transmitted","contribution_residual","linear_reference","truncation_unknown"};
                const auto& image=component<4 ? captured.value().contributions[component] : component==4 ? captured.value().linearReference : captured.value().truncationUnknown;
                const String id=options.sampleId+"/"+c.id+"/"+names[component];
                const String path="ground_truth/"+c.id+"/"+names[component]+".exr";
                Json description={{"role","ground_truth"},{"signal",{{"kind",names[component]},
                    {"unit",component==5 ? "mask" : measurementUnit},{"channels",image.channelNames},{"storage","float32"},{"transfer","linear"}}},
                    {"stage","response_psf_exposure_before_detector_state_and_noise"},{"coverage","all_samples"},
                    {"contribution_definition","first_camera_side_branch"},{"physical_tail","unknown_if_truncation_mask_is_set"},
                    {"provenance",{{"geometry",geometry}}}};
                auto written=session.WriteImage(path,id,image,description.dump());if(!written)throw std::runtime_error(written.error());
                truthRecord["products"].push_back({{"product_id",id},{"path",path}});
                if(component<4)truthRecord["contribution_products"].push_back(id);
                if(component==4)truthRecord["linear_reference_product"]=id;
                if(component==5)truthRecord["truncation_unknown_product"]=id;
            }
            for(const auto& [image,name,unit]:geometryImages) {
                const String id=options.sampleId+"/"+c.id+"/"+name;
                const String path="ground_truth/"+c.id+"/"+name+".exr";
                const Json description={{"role","ground_truth"},{"signal",{{"kind",name},{"unit",unit},
                    {"channels",image->channelNames},{"storage","float32"},{"transfer","linear"}}},
                    {"provenance",{{"geometry",geometry}}}};
                const auto wrote=session.WriteImage(path,id,*image,description.dump());
                if(!wrote) throw std::runtime_error(wrote.error());
                truthRecord["products"].push_back({{"product_id",id},{"path",path}});
            }
            const String idsId=options.sampleId+"/"+c.id+"/instance_id";
            const String idsPath="ground_truth/"+c.id+"/instance_id.exr";
            const Json idsDescription={{"role","ground_truth"},{"signal",{{"kind","instance_id"},
                {"unit","identifier"},{"channels",{"instance_id"}},{"storage","uint32"},{"transfer","identity"}}},
                {"provenance",{{"geometry",geometry}}}};
            const auto wroteIds=session.WriteUIntImage(idsPath,idsId,truth.value().instanceId,idsDescription.dump());
            if(!wroteIds) throw std::runtime_error(wroteIds.error());
            truthRecord["products"].push_back({{"product_id",idsId},{"path",idsPath}});
            manifest["ground_truth"].push_back(std::move(truthRecord));
            geometries.emplace(c.id,std::move(*truth));
            const std::pair<std::optional<camera::CameraProduct>*,const char*> products[]={
                {&images.bandMeasurement,"measurement"},{&images.rawDn,"raw_dn"},
                {&images.correctedDeviceSignal,"corrected"},{&images.apparentTemperature,"temperature"},
                {&images.cieLinearSrgb,"cie_linear_srgb"},{&images.tracedRadiance,"spectral"},{&images.display,"display"}};
            u64 acquisitionIndex=0;
            for(const auto& [image,name]:products) if(*image) { acquisitionIndex=(*image)->signal.acquisitionIndex;break; }
            const String acquisition=options.sampleId+"/"+c.id+"/"+std::to_string(acquisitionIndex);
            auto& actualEndpoints=endpoints[c.id];
            std::erase_if(actualEndpoints,[&](const auto& e){return e.acquisitionIndex!=acquisitionIndex;});
            auto portableSensor=c.sensor;
            if(!c.sensor.optics.knownPsfPath.empty()) {
                const String name="provenance/"+c.id+"/psf"+fs::path(c.sensor.optics.knownPsfPath).extension().string();
                auto staged=session.StagingPath(name);
                if(!staged) throw std::runtime_error(staged.error());
                fs::copy_file(fs::path(c.sensor.optics.knownPsfPath),fs::path(*staged),fs::copy_options::overwrite_existing);
                auto registered=session.RegisterFile(name,c.id+"/psf",R"({"role":"calibration_resource"})");
                if(!registered) throw std::runtime_error(registered.error());
                portableSensor.optics.knownPsfPath=name;
                portableSensor.optics.knownPsfSourcePath=name;
            }
            Json observation={{"camera_id",c.id},{"acquisition_id",acquisition},{"acquisition_index",acquisitionIndex},
                {"camera_to_rig",c.cameraToRig},{"rig_to_world",rig.rigToWorld},
                {"products",Json::array()},{"sensor_config_toml",CameraConfigToToml(portableSensor)}};
            observation["optical_assumptions"]={{"psf_spatial_model","shift_invariant_native_pixel_kernel"},
                {"psf_source",c.sensor.optics.knownPsfPath.empty() ? "authored_or_diffraction_approximation" : "bundled_kernel"},
                {"vignetting_model",c.sensor.optics.cosFourthVignetting ? "authored_perspective_cos_fourth" : "none"},
                {"projection_does_not_establish_optical_response_calibration",true}};
            Image lensValidity(pathGeometry.width,pathGeometry.height,1);
            lensValidity.channelNames={"lens_valid_centre"};
            for(u32 y=0;y<lensValidity.height;++y)for(u32 x=0;x<lensValidity.width;++x)
                lensValidity(x,y,0)=camera::UnprojectPixel(*pathGeometry.nativeProjection,{x+.5,y+.5}).valid ? 1.0f : 0.0f;
            const Json lensSignal={{"kind","lens_validity"},{"unit","mask"},
                {"storage","float32"},{"channels",lensValidity.channelNames},{"transfer","identity"}};
            const String lensId=options.sampleId+"/"+c.id+"/lens_validity";
            const String lensPath="observations/"+c.id+"/lens_validity.exr";
            auto lensWritten=session.WriteImage(lensPath,lensId,lensValidity,Json{
                {"role","observation"},{"signal",lensSignal},{"camera_id",c.id},{"acquisition_id",acquisition}}.dump());
            if(!lensWritten)throw std::runtime_error(lensWritten.error());
            observation["products"].push_back({{"product_id",lensId},{"path",lensPath},{"signal",lensSignal}});
            for(const auto& [image,name]:std::vector<std::pair<const Image*,String>>{
                {&captured.value().lensValidity,"lens_status"},{&captured.value().validSampleFraction,"valid_sample_fraction"}}) {
                const String id=options.sampleId+"/"+c.id+"/"+name;
                const String path="observations/"+c.id+"/"+name+".exr";
                const Json signal={{"kind",name},{"unit",name=="lens_status" ? "enum" : "fraction"},
                    {"storage","float32"},{"channels",image->channelNames},{"transfer","identity"}};
                const Json description={{"role","observation"},{"signal",signal},{"camera_id",c.id},{"acquisition_id",acquisition},
                    {"status_codes",{{"valid",0},{"outside_field",1},{"inverse_failed",2},{"outside_image",3}}}};
                auto written=session.WriteImage(path,id,*image,description.dump());if(!written)throw std::runtime_error(written.error());
                observation["products"].push_back({{"product_id",id},{"path",path},{"signal",signal}});
            }
            for(const auto& [image,name]:products) if(*image) {
                if(options.cancelled && options.cancelled()) throw std::runtime_error("fusion export cancelled");
                const String id=options.sampleId+"/"+c.id+"/"+name;
                const String path="observations/"+c.id+"/"+name+".exr";
                const String productAcquisition=(*image)->signal.acquisitionIndex==acquisitionIndex ? acquisition :
                    options.sampleId+"/"+c.id+"/"+std::to_string((*image)->signal.acquisitionIndex);
                const Json description={{"role","observation"},{"signal",Signal(**image)},
                    {"camera_id",c.id},{"acquisition_id",productAcquisition},
                    {"acquisition_kind",productAcquisition==acquisition ? "sensor_capture" : "independent_observer_render"}};
                const auto wrote=session.WriteImage(path,id,(*image)->image,description.dump());
                if(!wrote) throw std::runtime_error(wrote.error());
                observation["products"].push_back({{"product_id",id},{"path",path},{"signal",Signal(**image)},{"acquisition_id",productAcquisition}});
                // Prefer corrected device data, then raw/measurement; never default to display.
                if(String(name)=="corrected" || (!primaryProducts.contains(c.id) && String(name)!="display"))
                    primaryProducts[c.id]=id;
                if(options.rectify && ((*image)->signal.cfa==camera::CfaPattern::Mono ||
                   (*image)->signal.kind==camera::SignalKind::DisplaySrgb ||
                   (*image)->signal.kind==camera::SignalKind::CieLinearSrgb)) {
                    const auto native=*geometries.at(c.id).geometry.nativeProjection;
                    const auto target=RectifiedProjection(native,(*image)->image.width,(*image)->image.height);
                    Image mapping,valid;
                    auto rectified=Rectify((*image)->image,native,target,mapping,valid);
                    auto g=geometries.at(c.id).geometry;g.nativeProjection=target;
                    const auto rectGeometry=g.ToJson();
                    if(!rectGeometry) throw std::runtime_error(rectGeometry.error());
                    Json frozen=Json::parse(rectified.metadata.at("quantiloom_provenance"));
                    frozen["geometry"]=Json::parse(*rectGeometry);
                    frozen["processing"]={{"operation","undistort"},{"resampling","bilinear"},{"parent_product",id}};
                    rectified.metadata["quantiloom_provenance"]=frozen.dump();
                    const String rectId=id+"/rectified";
                    Json rectDescription=description;
                    rectDescription["parent_product"]=id;
                    rectDescription["processing"]={{"operation","undistort"},{"resampling","bilinear"}};
                    auto written=session.WriteImage("observations/"+c.id+"/"+name+"_rectified.exr",rectId,rectified,rectDescription.dump());
                    if(!written) throw std::runtime_error(written.error());
                    observation["products"].push_back({{"product_id",rectId},
                        {"path","observations/"+c.id+"/"+name+"_rectified.exr"},{"signal",Signal(**image)},{"parent_product",id},
                        {"acquisition_id",productAcquisition}});
                    if(!primaryProducts.contains(c.id+"/rectified")) {
                        primaryProducts[c.id+"/rectified"]=rectId;
                        for(const auto& [map,mapName]:std::vector<std::pair<Image*,String>>{
                                {&mapping,"rectification_map"},{&valid,"rectification_valid"}}) {
                            const Json mapDescription={{"role","ground_truth"},{"signal",{{"kind",mapName},
                                {"unit",mapName=="rectification_map" ? "native_pixels" : "mask"},
                                {"channels",map->channelNames},{"storage","float32"},{"transfer","linear"}}}};
                            written=session.WriteImage("ground_truth/"+c.id+"/"+mapName+".exr",
                                options.sampleId+"/"+c.id+"/"+mapName,*map,mapDescription.dump());
                            if(!written) throw std::runtime_error(written.error());
                        }
                        auto rectTruth=renderer.value()->CaptureGeometry(options.referenceTimeSeconds,&target);
                        if(!rectTruth) throw std::runtime_error(rectTruth.error());
                        for(size_t i=0;i<valid.data.size();++i) if(valid.data[i]==0)
                            rectTruth.value().validity.data[i]=2;
                        Json rectRecord={{"camera_id",c.id},{"raster_variant","rectified"},
                            {"geometry",Json::parse(*rectGeometry)},{"products",Json::array()},
                            {"sampling","instantaneous_pixel_centre"},{"instances",Json::parse(rectTruth.value().instancesJson)}};
                        const std::tuple<Image*,const char*,const char*> rectImages[]={
                            {&rectTruth.value().rayDistanceMeters,"ray_distance","m"},
                            {&rectTruth.value().cameraDepthMeters,"camera_z","m"},
                            {&rectTruth.value().worldPosition,"world_position","world_units"},
                            {&rectTruth.value().worldNormal,"geometric_normal","dimensionless"},
                            {&rectTruth.value().validity,"geometry_validity","enum"}};
                        for(const auto& [field,fieldName,unit]:rectImages) {
                            const String fieldId=options.sampleId+"/"+c.id+"/rectified/"+fieldName;
                            const String fieldPath="ground_truth/"+c.id+"/rectified/"+fieldName+".exr";
                            auto published=session.WriteImage(fieldPath,fieldId,*field,Json{
                                {"role","ground_truth"},{"signal",{{"kind",fieldName},{"unit",unit},
                                    {"channels",field->channelNames},{"storage","float32"},{"transfer","linear"}}},
                                {"provenance",{{"geometry",Json::parse(*rectGeometry)}}}}.dump());
                            if(!published)throw std::runtime_error(published.error());
                            rectRecord["products"].push_back({{"product_id",fieldId},{"path",fieldPath}});
                        }
                        const String rectIds=options.sampleId+"/"+c.id+"/rectified/instance_id";
                        const String rectIdsPath="ground_truth/"+c.id+"/rectified/instance_id.exr";
                        auto published=session.WriteUIntImage(rectIdsPath,rectIds,rectTruth.value().instanceId,Json{
                            {"role","ground_truth"},{"signal",{{"kind","instance_id"},{"unit","identifier"},
                                {"channels",{"instance_id"}},{"storage","uint32"},{"transfer","identity"}}},
                            {"provenance",{{"geometry",Json::parse(*rectGeometry)}}}}.dump());
                        if(!published)throw std::runtime_error(published.error());
                        rectRecord["products"].push_back({{"product_id",rectIds},{"path",rectIdsPath}});
                        manifest["ground_truth"].push_back(rectRecord);
                        geometries.emplace(c.id+"/rectified",std::move(*rectTruth));
                    }
                }
            }
            if(!primaryProducts.contains(c.id)) throw std::runtime_error("camera has no non-display observation product");
            manifest["observations"].push_back(std::move(observation));
            ++completed;
        }
        for(const auto& p:rig.pairs) for(u32 variant=0;variant<(options.rectify ? 2u : 1u);++variant) {
            if(options.onProgress) options.onProgress({completed,completed,p.sourceCamera,"correspondence"});
            if(options.cancelled && options.cancelled()) throw std::runtime_error("fusion export cancelled");
            if(!primaryProducts.contains(p.sourceCamera) || !primaryProducts.contains(p.targetCamera))
                throw std::runtime_error("pair references unknown camera");
            const String suffix=variant ? "/rectified" : "";
            if(variant && (!geometries.contains(p.sourceCamera+suffix) || !geometries.contains(p.targetCamera+suffix))) continue;
            const auto& source=geometries.at(p.sourceCamera+suffix);
            const auto& target=geometries.at(p.targetCamera+suffix);
            const auto& camera=target.geometry.camera;
            const auto& projection=*target.geometry.nativeProjection;
            Image coordinates(source.geometry.width,source.geometry.height,2);
            coordinates.channelNames={"target_x","target_y"};
            Image validity(source.geometry.width,source.geometry.height,1);
            validity.channelNames={"correspondence_validity"};
            Vector<glm::vec2> queries(source.instanceId.pixels.size(),glm::vec2(projection.cx,projection.cy));
            for(size_t i=0;i<queries.size();++i) {
                if(source.validity.data[i]!=1) { validity.data[i]=source.validity.data[i]>=3 ? 4.0f : 0.0f; continue; }
                const glm::dvec3 point(source.worldPosition.data[i*3],source.worldPosition.data[i*3+1],source.worldPosition.data[i*3+2]);
                const auto offset=point-glm::dvec3(camera.origin);
                const auto q=camera::ProjectDirection(projection,{glm::dot(offset,glm::dvec3(camera.right)),
                    -glm::dot(offset,glm::dvec3(camera.up)),glm::dot(offset,glm::dvec3(camera.forward))});
                if(!q.valid || q.pixel.x<0 || q.pixel.y<0 || q.pixel.x>=target.geometry.width || q.pixel.y>=target.geometry.height) {
                    validity.data[i]=2; continue;
                }
                if(variant) {
                    const auto ray=camera::UnprojectPixel(projection,q.pixel);
                    const auto native=camera::ProjectDirection(*geometries.at(p.targetCamera).geometry.nativeProjection,ray.direction);
                    if(!native.valid || !detail::HasRectificationSupport(
                       *geometries.at(p.targetCamera).geometry.nativeProjection,native.pixel,
                       target.geometry.width,target.geometry.height)) {
                        validity.data[i]=2;continue;
                    }
                    queries[i]=glm::vec2(native.pixel);
                } else queries[i]=glm::vec2(q.pixel);
                coordinates.data[i*2]=static_cast<f32>(q.pixel.x);
                coordinates.data[i*2+1]=static_cast<f32>(q.pixel.y);
                validity.data[i]=5;
            }
            auto targetConfig=RigCameraScene(scene,rig,p.targetCamera);
            if(!targetConfig) throw std::runtime_error(targetConfig.error());
            auto verifier=OfflineRenderer::Create(*targetConfig,init);
            if(!verifier) throw std::runtime_error(verifier.error());
            const auto targetHits=verifier.value()->QuerySurfaces(queries,options.referenceTimeSeconds);
            if(!targetHits) throw std::runtime_error(targetHits.error());
            for(size_t i=0;i<queries.size();++i) if(validity.data[i]==5) {
                const auto& hit=targetHits.value()[i];
                const glm::vec3 position(source.worldPosition.data[i*3],source.worldPosition.data[i*3+1],source.worldPosition.data[i*3+2]);
                const f64 magnitude=std::max({1.0,std::abs(static_cast<f64>(position.x)),
                    std::abs(static_cast<f64>(position.y)),std::abs(static_cast<f64>(position.z)),
                    static_cast<f64>(source.rayDistanceMeters.data[i])/source.geometry.worldUnitsToMeters});
                const f64 tolerance=std::max(1e-4,8*std::numeric_limits<f32>::epsilon()*magnitude*source.geometry.worldUnitsToMeters);
                validity.data[i]=hit.validity>=3 ? 4.0f :
                    hit.instanceId==source.instanceId.pixels[i] &&
                    glm::length(glm::dvec3(hit.worldPosition-position))*source.geometry.worldUnitsToMeters<=tolerance ? 1.0f : 3.0f;
            }
            const String stem=p.sourceCamera+"_to_"+p.targetCamera+(variant ? "_rectified" : "");
            String opticalId;
            {
                if(options.cancelled && options.cancelled())throw std::runtime_error("fusion export cancelled");
                const auto optical=MatchOpticalPaths(*verifier.value(),endpoints.at(p.sourceCamera),
                    endpoints.at(p.targetCamera),geometries.at(p.targetCamera).geometry,options.maxOpticalSourcePaths,options.cancelled);
                if(!optical)throw std::runtime_error(optical.error());
                Json mapping=Json::parse(*optical);
                if(variant) {
                    mapping["coordinate_variant"]="rectified";
                    mapping["native_parent_product"]=options.sampleId+"/pairs/"+p.sourceCamera+"_to_"+p.targetCamera+"/optical_paths";
                    const auto convert=[&](const Json& pixel,const String& id)->Json {
                        const auto ray=camera::UnprojectPixel(*geometries.at(id).geometry.nativeProjection,
                            {pixel[0].get<f64>(),pixel[1].get<f64>()});
                        if(!ray.valid)return nullptr;
                        const auto projected=camera::ProjectDirection(*geometries.at(id+suffix).geometry.nativeProjection,ray.direction);
                        if(!projected.valid || projected.pixel.x<0 || projected.pixel.y<0 ||
                            projected.pixel.x>=geometries.at(id+suffix).geometry.width || projected.pixel.y>=geometries.at(id+suffix).geometry.height)return nullptr;
                        const auto& np=*geometries.at(id).geometry.nativeProjection;
                        const auto at=camera::ProjectDirection(np,ray.direction);
                        if(!at.valid || !detail::HasRectificationSupport(np,at.pixel,
                            geometries.at(id).geometry.width,geometries.at(id).geometry.height))return nullptr;
                        return Json{projected.pixel.x,projected.pixel.y};
                    };
                    for(auto& row:mapping["rows"]) {
                        row["source_rectified_pixel"]=convert(row["source_native_pixel"],p.sourceCamera);
                        for(auto& match:row["matches"]) {
                            match["target_native_pixel"]=match["target_pixel"];
                            match["target_pixel"]=convert(match["target_native_pixel"],p.targetCamera);
                            match["rectification_valid"]=!row["source_rectified_pixel"].is_null() && !match["target_pixel"].is_null();
                        }
                    }
                } else mapping["coordinate_variant"]="native";
                opticalId=options.sampleId+"/pairs/"+stem+"/optical_paths";
                const String path="ground_truth/pairs/"+stem+"_optical.json";
                auto staged=session.StagingPath(path);if(!staged)throw std::runtime_error(staged.error());
                WriteText(*staged,mapping.dump());
                const auto registered=session.RegisterFile(path,opticalId,R"({"role":"path_correspondence"})");
                if(!registered)throw std::runtime_error(registered.error());
            }
            const String coordinatesId=options.sampleId+"/pairs/"+stem+"/coordinates";
            const String validityId=options.sampleId+"/pairs/"+stem+"/validity";
            for(const auto& [image,id,name]:std::vector<std::tuple<Image*,String,String>>{
                    {&coordinates,coordinatesId,"coordinates"},{&validity,validityId,"validity"}}) {
                const Json description={{"role","ground_truth"},{"signal",{{"kind",name=="coordinates" ? "pixel_correspondence" : "correspondence_validity"},
                    {"unit",name=="coordinates" ? "target_pixels" : "enum"},{"channels",image->channelNames},{"storage","float32"},{"transfer","linear"}}}};
                auto written=session.WriteImage("ground_truth/pairs/"+stem+"_"+name+".exr",id,*image,description.dump());
                if(!written) throw std::runtime_error(written.error());
            }
            manifest["pairs"].push_back({{"source_product",primaryProducts.at(p.sourceCamera+suffix)},
                {"target_product",primaryProducts.at(p.targetCamera+suffix)},
                {"reference_time_s",options.referenceTimeSeconds},{"correspondence_status","opaque_geometry_verified"},
                {"optical_correspondence_product",opticalId},
                {"coordinates_product",coordinatesId},{"validity_product",validityId},
                {"mapping",variant ? "source_to_target_absolute_rectified_pixel_centres" : "source_to_target_absolute_native_pixel_centres"},
                {"validity_codes",{{"source_invalid",0},{"valid",1},{"outside_target",2},{"occluded",3},{"nonopaque",4},{"unresolved",5}}}});
        }
        auto staged=session.StagingPath(manifestName);
        if(!staged) throw std::runtime_error(staged.error());
        WriteText(*staged,manifest.dump(2)+"\n");
        auto registered=session.RegisterFile(manifestName,"manifest",R"({"role":"manifest"})");
        if(!registered) throw std::runtime_error(registered.error());
        if(options.onProgress) options.onProgress({completed,completed,"","ready_to_publish"});
        if(options.cancelled && options.cancelled()) throw std::runtime_error("fusion export cancelled");
        if(options.onProgress) options.onProgress({completed,completed,"","publish"});
        auto committed=session.Commit();
        if(!committed) throw std::runtime_error(committed.error());
        return FusionExportResult{(directory/session.SidecarName()).string(),
            (directory/manifestName).string(),manifest.dump(2)};
    } catch(const std::exception& e) { return Result<FusionExportResult,String>::Err(e.what()); }
}

Result<FusionExportResult,String> FusionExportJob::RunFile(const String& path,bool dryRun) {
    auto job=Config::Load(fs::path(path));
    if(!job) return Result<FusionExportResult,String>::Err(job.error());
    const auto base=fs::absolute(fs::path(path)).parent_path();
    auto rig=ParseRigConfig(*job,base.string());
    if(!rig) return Result<FusionExportResult,String>::Err(rig.error());
    auto scene=Config::Load(base/job.value().GetString("fusion.scene_config"));
    if(!scene) return Result<FusionExportResult,String>::Err(scene.error());
    FusionExportOptions options;
    options.sampleId=job.value().GetString("fusion.sample_id","sample");
    options.outputDirectory=(base/job.value().GetString("fusion.output_directory","fusion-output")).string();
    options.referenceTimeSeconds=job.value().GetDouble("fusion.time_s",0);
    options.rectify=job.value().GetBool("fusion.rectify",false);
    options.maxRecordedRays=job.value().GetUInt("fusion.max_recorded_rays",4096);
    options.maxOpticalSourcePaths=job.value().GetUInt("fusion.max_optical_source_paths",64);
    options.dryRun=dryRun;
    options.renderer.baseDir=(base/fs::path(job.value().GetString("fusion.scene_config"))).parent_path().string();
    return Run(*scene,*rig,options);
}
} // namespace quantiloom::dataset
