#!/usr/bin/env python3
"""Validate a standalone fusion sample without renderer or SDK dependencies."""
import argparse
import pathlib
import sys
import json
import struct
import math
import dataset_verify as exports


def verify(path, inspect_exr=False):
    path = pathlib.Path(path)
    manifest = exports.load(path)
    if manifest['schema'] != 'quantiloom.fusion.sample' or manifest['schema_version'] not in (1,2):
        raise ValueError('unsupported fusion sample schema')
    name = pathlib.PurePosixPath(manifest['record_path'])
    if len(name.parts) != 1 or name.name in ('.', '..') or ':' in str(name) or '\\' in str(name):
        raise ValueError('invalid export record reference')
    record_path = path.parent / name.name
    integrity = exports.verify(record_path, inspect_exr)
    record = exports.load(record_path)
    if record['schema_version'] != 2 or record['record_id'] != manifest['record_id']:
        raise ValueError('fusion record identity/version mismatch')
    products = {p['product_id']: p for p in record['products']}
    own = products.get('manifest')
    if own is None or own['path'] != path.name:
        raise ValueError('manifest is not a managed artifact')
    cameras, observation_ids, acquisitions = set(), set(), set()
    for camera in manifest['observations']:
        identity = camera['camera_id']
        if identity in cameras:
            raise ValueError('duplicate camera identity')
        cameras.add(identity)
        acquisition = camera['acquisition_id']
        if acquisition in acquisitions:
            raise ValueError('duplicate acquisition identity')
        acquisitions.add(acquisition)
        for ref in camera['products']:
            product = products[ref['product_id']]
            d = product['description']
            if product['path'] != ref['path'] or d['role'] != 'observation':
                raise ValueError('observation reference mismatch')
            if d['camera_id'] != identity or d['acquisition_id'] != ref.get('acquisition_id', acquisition) or d['signal'] != ref['signal']:
                raise ValueError('observation semantics/identity mismatch')
            if ref['product_id'] in observation_ids:
                raise ValueError('duplicate observation reference')
            observation_ids.add(ref['product_id'])
            if 'parent_product' in ref and ref['parent_product'] not in products:
                raise ValueError('missing parent product')
    if manifest['reference_camera'] not in cameras:
        raise ValueError('missing reference camera')
    for truth in manifest['ground_truth']:
        if truth['camera_id'] not in cameras:
            raise ValueError('ground truth references unknown camera')
        for ref in truth['products']:
            p = products[ref['product_id']]
            if p['path'] != ref['path'] or p['description']['role'] != 'ground_truth':
                raise ValueError('ground truth reference mismatch')
        for ref in truth.get('path_chunks', []):
            p = products[ref['product_id']]
            if p['path'] != ref['path'] or p['description'] != ref['description'] or p['description']['role'] != 'path_truth':
                raise ValueError('path record reference mismatch')
            check_paths(path.parent / p['path'], p['description'])
        if manifest['schema_version']==2 and truth.get('raster_variant','native')=='native':
            ids=truth['contribution_products']
            if len(ids)!=4 or len(set(ids))!=4:raise ValueError('expected four distinct contribution products')
            reference=products[truth['linear_reference_product']]
            for identity in ids:
                component=products[identity]
                if component['description']['signal']['unit']!=reference['description']['signal']['unit']:
                    raise ValueError('contribution unit mismatch')
                if component['description']['coverage']!='all_samples':raise ValueError('partial reconstruction coverage')
            if inspect_exr:
                import OpenEXR
                import numpy as np
                def raster(product):
                    with OpenEXR.File(str(path.parent/product['path']),separate_channels=True) as f:
                        return np.stack([f.channels()[name].pixels for name in product['description']['channel_names']],axis=-1).astype(np.float64)
                expected=raster(reference)
                total=sum(raster(products[identity]) for identity in ids)
                if not np.all(np.isfinite(total)) or not np.all(np.isfinite(expected)) or np.any(np.abs(total-expected)>1e-5*np.maximum(np.abs(expected),1e-20)):
                    raise ValueError('contributions do not reconstruct independent linear reference')
    for pair in manifest['pairs']:
        for key in ('source_product', 'target_product'):
            if pair[key] not in observation_ids:
                raise ValueError('pair references a non-observation product')
        for key in ('coordinates_product', 'validity_product'):
            if products[pair[key]]['description']['role'] != 'ground_truth':
                raise ValueError('pair truth has the wrong role')
        if pair['reference_time_s'] != manifest['reference_time_s']:
            raise ValueError('pair reference time mismatch')
        optical = pair.get('optical_correspondence_product')
        if optical:
            p = products[optical]
            if p['description']['role'] != 'path_correspondence':
                raise ValueError('wrong optical correspondence role')
            mapping = exports.load(path.parent / p['path'])
            if mapping['complete_solution_set'] or not mapping['unmatched_is_unresolved']:
                raise ValueError('unsupported optical completeness claim')
            for row in mapping['rows']:
                if row['source_path_product'] not in products:
                    raise ValueError('optical mapping references missing path records')
                for match in row['matches']:
                    if match.get('rectification_valid') is False:
                        continue
                    if not all(math.isfinite(x) for x in match['target_pixel']) or not math.isfinite(match['forward_residual_m']) or match['forward_residual_m'] < 0:
                        raise ValueError('invalid optical correspondence')
    return dict(integrity, checks='fusion_contract_and_export_integrity',
                camera_count=len(cameras), pair_count=len(manifest['pairs']))


def check_paths(path, description):
    data = path.read_bytes()
    if description['schema_version']==2:
        columns=path_columns(data,description)
        identities=columns['ray_identity'];components=columns['components'];radiance=columns['radiance']
        slots=description['slots_per_ray']
        for i,identity in enumerate(identities):
            pixel,sample,flags,depth=identity
            if not depth & 0x80000000:continue
            if pixel>=description['native_width']*description['native_height'] or sample>=description['spp'] or depth&0x7fffffff>=slots:
                raise ValueError('invalid column ray identity')
            values=components[i]+radiance[i]
            if not all(math.isfinite(x) for x in values) or abs(sum(components[i])-radiance[i][0])>1e-5*max(1e-20,abs(radiance[i][0])):
                raise ValueError('column contributions do not reconstruct ray radiance')
            for d in range((depth&0x7fffffff)+1):
                row=i*slots+d
                kind,node,primitive,route=columns['vertex_identity'][row]
                if kind and (kind not in (1,2,3,5) or node==0 or route>2):raise ValueError('invalid column vertex identity')
                for name in ('position_wavelength','normal_distance','outgoing_coefficient','medium_segment','vertex_radiance_terms'):
                    if not all(math.isfinite(x) for x in columns[name][row]):raise ValueError('nonfinite path column')
            reconstructed=0.0
            for d in reversed(range((depth&0x7fffffff)+1)):
                row=i*slots+d;surface,bulk,weight,residual=columns['vertex_radiance_terms'][row]
                transmittance=columns['medium_segment'][row][3]
                reconstructed=(weight*reconstructed+surface+residual)*transmittance+bulk
            if abs(reconstructed-radiance[i][0])>1e-5*max(1e-20,abs(radiance[i][0])):
                raise ValueError('vertex terms do not reconstruct independently recorded ray radiance')
        return
    header = struct.unpack_from('<32I', data)
    enabled, stride, count, slots, width, height, spp, flags = header[:8]
    if (stride, count, slots, width, height, spp, flags) != tuple(description[k] for k in
            ('ray_stride', 'stored_rays', 'slots_per_ray', 'native_width', 'native_height', 'spp', 'diagnostic_flags')):
        raise ValueError('path header disagrees with description')
    if not enabled or not 0 < stride or not 0 < slots <= 9 or len(data) != 128 + count * (32 + slots * 80):
        raise ValueError('invalid path record dimensions')
    if flags & 2:
        raise ValueError('ambiguous medium boundary in path data')
    for i in range(count):
        components = struct.unpack_from('<4f', data, 128+i*32)
        pixel, sample, ray_flags, depth = struct.unpack_from('<4I', data, 128+i*32+16)
        if not depth & 0x80000000:
            continue
        if pixel >= width*height or sample >= spp or depth & 0x7fffffff >= slots:
            raise ValueError('invalid path ray identity/depth')
        if not all(math.isfinite(x) for x in components) or abs(sum(components[:3])-components[3]) > 1e-5*max(1.0, abs(components[3])):
            raise ValueError('path contributions do not reconstruct radiance')
        for d in range((depth & 0x7fffffff)+1):
            base=128+count*32+(i*slots+d)*80
            kind,node,primitive,route=struct.unpack_from('<4I',data,base+32)
            if kind == 0:
                continue  # sky endpoint has no finite surface
            if kind not in (1,2,3,5) or node == 0 or route > 2:
                raise ValueError('invalid path vertex identity')
            values=struct.unpack_from('<8f',data,base)+struct.unpack_from('<8f',data,base+48)
            if not all(math.isfinite(x) for x in values):
                raise ValueError('non-finite path vertex')


def path_columns(data,description):
    if description.get('byte_order')!='little_endian':raise ValueError('unsupported path byte order')
    if description.get('diagnostic_flags',0)&(2|16):raise ValueError('invalid quantitative path diagnostics')
    if not 0<description['slots_per_ray']<=9 or description['ray_stride']<=0:raise ValueError('invalid path dimensions')
    count=description['stored_rays'];slots=description['slots_per_ray']
    expected={'components':('float32',count,4),'radiance':('float32',count,1),'ray_identity':('uint32',count,4),
              'native_pixel':('float32',count,2),'position_wavelength':('float32',count*slots,4),
              'normal_distance':('float32',count*slots,4),'vertex_identity':('uint32',count*slots,4),
              'outgoing_coefficient':('float32',count*slots,4),'medium_segment':('float32',count*slots,4),
              'vertex_radiance_terms':('float32',count*slots,4)}
    result={};offset=0
    for column in description['columns']:
        name=column['name'];spec=(column['type'],column['rows'],column['width'])
        if name in result or name not in expected or spec!=expected[name]:raise ValueError('invalid path column schema')
        size=column['rows']*column['width']*4
        if column['offset_bytes']!=offset or column['size_bytes']!=size or offset+size>len(data):raise ValueError('invalid column extent')
        fmt='<'+('f' if column['type']=='float32' else 'I')*column['width']
        result[name]=[struct.unpack_from(fmt,data,offset+r*column['width']*4) for r in range(column['rows'])]
        offset+=size
    if offset!=len(data) or set(result)!=set(expected):raise ValueError('incomplete path columns')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=pathlib.Path)
    parser.add_argument('--inspect-exr', action='store_true')
    args = parser.parse_args()
    try:
        result = verify(args.manifest, args.inspect_exr)
        print(json.dumps(result, indent=2)); return 0
    except (KeyError, TypeError, ValueError, OSError) as error:
        print(json.dumps({'valid': False, 'error': str(error)})); return 1


if __name__ == '__main__':
    sys.exit(main())
