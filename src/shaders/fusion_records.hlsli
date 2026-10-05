#ifndef QL_FUSION_RECORDS_HLSLI
#define QL_FUSION_RECORDS_HLSLI
[[vk::binding(36,0)]] RWByteAddressBuffer fusionRecords;
void FusionAccumulate(uint2 pixel,uint2 size,uint sample,float4 values,float total,float valid,uint flags) {
    const uint base=fusionRecords.Load(44);
    if(base==0)return;
    const uint address=base+(pixel.y*size.x+pixel.x)*32;
    const float weight=1.0/float(sample+1);
    const float4 previous=sample==0 ? 0 : asfloat(fusionRecords.Load4(address));
    const float2 old=sample==0 ? 0 : asfloat(fusionRecords.Load2(address+16));
    fusionRecords.Store4(address,asuint(previous+(values-previous)*weight));
    fusionRecords.Store2(address+16,asuint(old+(float2(total,valid)-old)*weight));
    const uint prior=fusionRecords.Load(address+24);
    fusionRecords.Store(address+24,prior|flags);
    if(flags!=0){uint unused;fusionRecords.InterlockedOr(28,flags,unused);}
}
void FusionMarkUnknownTail() {
    const uint base=fusionRecords.Load(44);
    if(base==0)return;
    const uint2 pixel=DispatchRaysIndex().xy,size=DispatchRaysDimensions().xy;
    uint unused;
    fusionRecords.InterlockedOr(base+(pixel.y*size.x+pixel.x)*32+24,1,unused);
    fusionRecords.InterlockedOr(28,1,unused);
}
uint FusionVertexAddress(Payload p) {
    const uint rays=fusionRecords.Load(8),slots=fusionRecords.Load(12);
    return 128+rays*32+(p.fusionPathId*slots+p.depth)*80;
}
void FusionRecordVertex(Payload p,float3 hit,float3 normal,uint nodeId,uint primitiveId,uint kind) {
    if(p.fusionPathId==0xFFFFFFFFu || p.depth>=fusionRecords.Load(12)) return;
    uint address=FusionVertexAddress(p);
    fusionRecords.Store4(address,asuint(float4(hit,p.heroLambda!=0 ? abs(p.heroLambda) : pushConsts.camera.wavelength_nm)));
    fusionRecords.Store4(address+16,asuint(float4(normal,RayTCurrent()+p.fusionSegmentOffset)));
    fusionRecords.Store4(address+32,uint4(kind,nodeId,primitiveId,p.fusionRoute));
    fusionRecords.Store4(address+64,asuint(float4(1,1,0,1)));
}
void FusionRecordInterface(Payload p,float3 outgoing,float weight,float n1,float n2) {
    if(p.fusionPathId==0xFFFFFFFFu || p.depth>=fusionRecords.Load(12)) return;
    uint address=FusionVertexAddress(p);
    fusionRecords.Store4(address+48,asuint(float4(outgoing,weight)));
    fusionRecords.Store2(address+64,asuint(float2(n1,n2)));
}
void FusionRecordSegment(Payload p,float sigma,float transmittance) {
    if(p.fusionPathId==0xFFFFFFFFu || p.depth>=fusionRecords.Load(12)) return;
    fusionRecords.Store2(FusionVertexAddress(p)+72,asuint(float2(sigma,transmittance)));
}
void FusionRecordRadianceTerm(Payload p,uint term,float value) {
    const uint base=fusionRecords.Load(56);
    if(base==0 || p.fusionPathId==0xFFFFFFFFu || p.depth>=fusionRecords.Load(12))return;
    fusionRecords.Store(base+(p.fusionPathId*fusionRecords.Load(12)+p.depth)*16+term*4,asuint(value));
}
void FusionRecordSky(Payload p) {
    if(p.fusionPathId==0xFFFFFFFFu || p.depth>=fusionRecords.Load(12))return;
    fusionRecords.Store4(FusionVertexAddress(p)+64,asuint(float4(1,1,0,1)));
    FusionRecordRadianceTerm(p,0,p.radiance.x);
}
#endif
