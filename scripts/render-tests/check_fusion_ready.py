#!/usr/bin/env python3
"""Fusion contract and native-camera geometry gate; all output is ASCII."""
import json
import pathlib
import subprocess
import sys
import math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import fusion_verify


def run():
    import OpenEXR
    import numpy as np
    root=pathlib.Path(__file__).resolve().parents[2]
    cli=root/'build/src/app/Release/Quantiloom.exe'
    job=root/'docs/dataset/fusion/example.toml'
    render=subprocess.run([str(cli),'fusion-export',str(job)],cwd=root,capture_output=True,text=True,errors='replace')
    if render.returncode:
        if 'no suitable' in render.stdout.lower() or 'no vulkan' in render.stdout.lower():
            print('SKIP fusion gate: no Vulkan ray tracing device');return 3
        raise ValueError('fusion render failed: '+render.stderr[-1000:])
    path=root/'build/fusion-example/plate.manifest.json'
    report=fusion_verify.verify(path,True)
    manifest=json.loads(path.read_text())
    record=json.loads((path.parent/manifest['record_path']).read_text())
    products={p['product_id']:p for p in record['products']}
    def raster(product_id):
        with OpenEXR.File(str(path.parent/products[product_id]['path']),separate_channels=True) as image:
            channels=products[product_id]['description']['channel_names']
            return np.stack([image.channels()[name].pixels for name in channels],axis=-1)
    checked=0
    for truth in manifest['ground_truth']:
        g=truth['geometry'];p=g['distortion'];k=g['intrinsics']
        world=raster(next(p['product_id'] for p in truth['products'] if p['product_id'].endswith('/world_position')))
        mask=raster(next(p['product_id'] for p in truth['products'] if p['product_id'].endswith('/geometry_validity')))[...,0]==1
        if not mask.any():raise ValueError('geometry fixture has no opaque hits')
        points=world[mask].astype(np.float64)
        transform=np.array(g['world_to_camera']).reshape(4,4)
        camera=points@transform[:3,:3].T+transform[:3,3]
        x,y=camera[:,0]/camera[:,2],camera[:,1]/camera[:,2]
        if p['model']=='brown_conrady':
            k1,k2,p1,p2,k3=p['coefficients'];r=x*x+y*y;a=1+k1*r+k2*r*r+k3*r*r*r
            x,y=x*a+2*p1*x*y+p2*(r+2*x*x),y*a+p1*(r+2*y*y)+2*p2*x*y
        elif p['model']=='fisheye':
            r=np.sqrt(x*x+y*y);theta=np.arctan(r);coeff=p['coefficients']
            td=theta*(1+sum(coeff[i]*theta**(2*i+2) for i in range(4)))
            scale=np.divide(td,r,out=np.ones_like(r),where=r>0);x,y=x*scale,y*scale
        yy,xx=np.nonzero(mask)
        error=np.maximum(np.abs(k[0]*x+k[2]-(xx+.5)),np.abs(k[4]*y+k[5]-(yy+.5)))
        if error.max()>.01:raise ValueError('GPU projection exceeds 0.01 pixel tolerance')
        checked+=int(mask.sum())
    for pair in manifest['pairs']:
        mask=raster(pair['validity_product'])[...,0]
        if not np.any(mask==1):raise ValueError('pair fixture has no verified correspondences')
    print('Fusion gate passed: %d cameras, %d pairs, %d checked opaque pixels' % (report['camera_count'],report['pair_count'],checked))
    check_transmission(root,cli)
    from fusion_rectification_reference import check
    check(root,cli)
    return 0


def check_transmission(root,cli):
    import struct
    sys.path.insert(0,str(root/'scripts/physics-audit'))
    import harness
    job=root/'docs/dataset/fusion/transmission_job.toml'
    render=subprocess.run([str(cli),'fusion-export',str(job)],cwd=root,capture_output=True,text=True,errors='replace')
    if render.returncode:raise ValueError('transmission render failed: '+render.stderr[-1000:])
    root=root/'build/fusion-transmission'
    manifest=json.loads((root/'glass.manifest.json').read_text())
    fusion_verify.verify(root/'glass.manifest.json',True)
    r,t,a=harness.homogeneous_slab_coefficients(1.5,10,.2)
    max_error=0
    for truth in manifest['ground_truth']:
        for chunk in truth['path_chunks']:
            data=(root/chunk['path']).read_bytes();d=chunk['description'];n=d['stored_rays']
            measured=sum(struct.unpack_from('<4f',data,128+i*32)[3] for i in range(n))/n
            wavelength=d['wavelength_nm']
            expected=t*harness.planck_blackbody(350,wavelength)+a*harness.planck_blackbody(300,wavelength)+r*harness.planck_blackbody(260,wavelength)
            error=abs(measured/expected-1);max_error=max(max_error,error)
            if error>.01:raise ValueError('slab radiance error exceeds 1 percent')
            if not any(struct.unpack_from('<4I',data,128+n*32+(i*9+2)*80+32)[1]==2 for i in range(n)):
                raise ValueError('no real target behind glass in the trace records')
    optical=json.loads((root/'ground_truth/pairs/offset_to_reference_optical.json').read_text())
    verified=sum(bool(row['matches']) for row in optical['rows'])
    if verified<10:raise ValueError('refractive correspondence fixture has too few verified paths')
    print('Transmission gate passed: slab error %.4f percent; %d verified source paths' % (100*max_error,verified))
    for case in ('curved', 'thin', 'nested'):
        job=root.parents[1]/'docs/dataset/fusion'/('%s_transmission_job.toml' % case)
        repo=root.parents[1]
        render=subprocess.run([str(cli),'fusion-export',str(job)],cwd=repo,capture_output=True,text=True,errors='replace')
        if render.returncode:raise ValueError('%s transmission failed: %s' % (case,render.stderr[-1000:]))
        sample=repo/'build'/('fusion-'+case)/(case+'.manifest.json')
        fusion_verify.verify(sample,True)
        mapping=json.loads((sample.parent/'ground_truth/pairs/offset_to_reference_optical.json').read_text())
        matched=sum(bool(r['matches']) for r in mapping['rows'])
        if matched<10:raise ValueError('%s has too few verified optical paths' % case)
        print('%s optical gate passed: %d verified source paths' % (case,matched))


if __name__=='__main__':
    try:sys.exit(run())
    except Exception as error:
        print('Fusion gate failed: '+str(error));sys.exit(1)
