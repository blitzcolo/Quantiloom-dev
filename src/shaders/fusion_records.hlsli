#ifndef QL_FUSION_RECORDS_HLSLI
#define QL_FUSION_RECORDS_HLSLI
[[vk::binding(36,0)]] RWByteAddressBuffer fusionRecords;
uint FusionVertexAddress(Payload p) {
    const uint rays=fusionRecords.Load(8),slots=fusionRecords.Load(12);
    return 128+rays*32+(p.fusionPathId*slots+p.depth)*80;
}
void FusionRecordVertex(Payload p,float3 hit,float3 normal,uint nodeId,uint primitiveId,uint kind) {
    if(p.fusionPathId==0xFFFFFFFFu || p.depth>=fusionRecords.Load(12)) return;
    uint address=FusionVertexAddress(p);
    fusionRecords.Store4(address,asuint(float4(hit,p.heroLambda!=0 ? abs(p.heroLambda) : pushConsts.camera.wavelength_nm)));
    fusionRecords.Store4(address+16,asuint(float4(normal,RayTCurrent())));
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
#endif
