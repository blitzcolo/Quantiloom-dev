#!/usr/bin/env python3
"""Independent export-integrity verifier. No renderer or SDK imports.

Python standard library checks JSON, managed hashes, PNG summaries and camera
matrices. --inspect-exr additionally requires OpenEXR and numpy. Resource and
execution equivalence are deliberately separate from export integrity.
"""
import argparse
import hashlib
import json
import math
import pathlib
import re
import struct
import sys
import zlib


def unique_object(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def load(path):
    def reject(value):
        raise ValueError(f'non-finite JSON number: {value}')

    record = json.loads(path.read_text(encoding='utf-8'),
                        object_pairs_hook=unique_object, parse_constant=reject)

    def finite(value):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError('non-finite JSON number')
        if isinstance(value, dict):
            for item in value.values():
                finite(item)
        elif isinstance(value, list):
            for item in value:
                finite(item)

    finite(record)
    return record


def digest(path):
    with path.open('rb') as stream:
        hasher = hashlib.sha256()
        for chunk in iter(lambda: stream.read(65536), b''):
            hasher.update(chunk)
        return hasher.hexdigest()


def check_geometry(g):
    """Validate one product's embedded instantaneous-geometry block."""
    if g['version'] not in (1, 2):
        raise ValueError(f"invalid geometry version: {g['version']!r}")
    if not (type(g['width']) is int and g['width'] > 0):
        raise ValueError(f"invalid geometry width: {g['width']!r}")
    if not (type(g['height']) is int and g['height'] > 0):
        raise ValueError(f"invalid geometry height: {g['height']!r}")
    if not g['kind'] == 'instantaneous_geometry':
        raise ValueError(f"invalid geometry kind: {g['kind']!r}")
    if not g['camera_axes'] == 'right_down_forward':
        raise ValueError(f"invalid geometry camera_axes: {g['camera_axes']!r}")
    if not g['pixel_origin'] == 'top_left':
        raise ValueError(f"invalid geometry pixel_origin: {g['pixel_origin']!r}")
    if not g['pixel_center_offset'] == [0.5, 0.5]:
        raise ValueError(
            f"invalid geometry pixel_center_offset: {g['pixel_center_offset']!r}")
    if not g['matrix_order'] == 'row_major':
        raise ValueError(f"invalid geometry matrix_order: {g['matrix_order']!r}")
    if not math.isfinite(g['reference_time_s']):
        raise ValueError(
            f"invalid geometry reference_time_s: {g['reference_time_s']!r}")
    if not (math.isfinite(g['world_units_to_meters']) and g['world_units_to_meters'] > 0):
        raise ValueError(
            f"invalid geometry world_units_to_meters: {g['world_units_to_meters']!r}")
    world, camera = (g['world_to_camera'], g['camera_to_world'])
    if not len(world) == len(camera) == 16:
        raise ValueError('camera matrices must both have 16 elements')
    if not all((math.isfinite(x) for x in world + camera)):
        raise ValueError('non-finite camera matrix element')
    for row in range(4):
        for col in range(4):
            value = sum((world[row * 4 + k] * camera[k * 4 + col] for k in range(4)))
            if not abs(value - (1 if row == col else 0)) < 1e-05:
                raise ValueError('camera matrices are not inverses')
    if g['projection'] == 'perspective':
        k = g['intrinsics']
        if not (len(k) == 9 and all((math.isfinite(x) for x in k))):
            raise ValueError(f'invalid perspective intrinsics: {k!r}')
        if not (k[0] > 0 and k[4] > 0 and (k[8] == 1)):
            raise ValueError(f'invalid perspective focal/normalization: {k!r}')
        if not k[1] == k[3] == k[6] == k[7] == 0:
            raise ValueError(f'non-canonical perspective intrinsics: {k!r}')
        if g['version'] == 1 and not (k[2] == g['width'] / 2 and k[5] == g['height'] / 2):
            raise ValueError('v1 principal point is not the image center')
        if g['version'] == 2:
            model = g['camera_model']
            distortion = g['distortion']
            if model not in ('pinhole', 'brown_conrady', 'fisheye') or distortion['model'] != model:
                raise ValueError(f'invalid native projection model: {model!r}')
            coefficients = distortion['coefficients']
            if len(coefficients) != 5 or not all(math.isfinite(x) for x in coefficients):
                raise ValueError(f'invalid native projection coefficients: {coefficients!r}')
            if model == 'pinhole' and any(coefficients):
                raise ValueError('pinhole carries distortion')
            if model == 'fisheye':
                theta = distortion['max_theta_radians']
                if not 0 < theta < math.pi / 2 or coefficients[4] != 0:
                    raise ValueError(f'invalid fisheye field: theta={theta!r}')
                for i in range(1025):
                    t = theta * i / 1024
                    if 1 + sum((2 * j + 3) * coefficients[j] * t ** (2 * j + 2)
                               for j in range(4)) <= 1e-8:
                        raise ValueError('non-monotone fisheye')
    else:
        if not (g['projection'] == 'orthographic' and g['intrinsics'] is None):
            raise ValueError(f"invalid orthographic projection: {g['projection']!r}")
        if not (g['film_height_world_units'] > 0 and g['film_width_world_units'] > 0):
            raise ValueError('invalid orthographic film extent')


def png_summary(path):
    summary = {}
    with path.open('rb') as stream:
        if not stream.read(8) == b'\x89PNG\r\n\x1a\n':
            raise ValueError('invalid PNG signature')
        while True:
            header = stream.read(8)
            if not len(header) == 8:
                raise ValueError('truncated PNG chunk header')
            size, kind = struct.unpack('>I4s', header)
            crc = zlib.crc32(kind)
            remaining = size
            data = bytearray()
            while remaining:
                part = stream.read(min(remaining, 65536))
                if not part:
                    raise ValueError('truncated PNG data')
                crc = zlib.crc32(part, crc)
                if kind in (b'tEXt', b'iTXt'):
                    if not size < 1024 * 1024:
                        raise ValueError('oversized PNG summary')
                    data.extend(part)
                remaining -= len(part)
            trailer = stream.read(4)
            if not (len(trailer) == 4 and struct.unpack('>I', trailer)[0] == crc):
                raise ValueError(f'PNG chunk CRC mismatch: {kind!r}')
            if kind in (b'tEXt', b'iTXt'):
                key, value = bytes(data).split(b'\x00', 1)
                if kind == b'iTXt':
                    if not value[:2] == b'\x00\x00':
                        raise ValueError('unsupported compressed PNG summary')
                    _, _, value = value[2:].split(b'\x00', 2)
                key = key.decode('latin1')
                if key in summary:
                    raise ValueError('duplicate PNG summary key')
                summary[key] = value.decode('utf-8')
            if kind == b'IEND':
                if not (size == 0 and (not stream.read(1))):
                    raise ValueError('invalid PNG end')
                return summary


def validate_record_shape(record):
    required = {'schema', 'schema_version', 'record_id', 'state', 'capture_status',
                'pairing_status', 'provenance', 'replay', 'products'}
    if not isinstance(record, dict) or not required.issubset(record):
        raise ValueError('missing required record fields')
    if set(record) - required - {'error'}:
        raise ValueError('unknown record field')
    if 'error' in record and not isinstance(record['error'], str):
        raise ValueError('error must be a string')
    if not isinstance(record['replay'], dict):
        raise ValueError('replay must be an object')
    if not isinstance(record['products'], list) or not record['products']:
        raise ValueError('record has no products')
    for product in record['products']:
        if not isinstance(product, dict) or not {'path', 'sha256', 'product_id',
                                                 'size_bytes', 'description'}.issubset(product):
            raise ValueError('missing required product fields')
        if type(product['size_bytes']) is not int or product['size_bytes'] < 0:
            raise ValueError(f"invalid product size: {product['size_bytes']!r}")
        if not isinstance(product['description'], dict):
            raise ValueError('product description must be an object')
        if record['schema_version'] == 2:
            d = product['description']
            if not isinstance(d.get('role'), str):
                raise ValueError('v2 product requires role')
            if d['role'] in ('observation', 'ground_truth'):
                signal = d['signal']
                if not all(isinstance(signal.get(k), str) and signal[k]
                           for k in ('kind', 'unit', 'storage', 'transfer')):
                    raise ValueError('missing v2 signal semantics')
                if signal['storage'] not in ('float32', 'uint32') \
                        or signal['channels'] != d['channel_names']:
                    raise ValueError('signal storage/channels mismatch')
                if signal['kind'] == 'instance_id' and signal['storage'] != 'uint32':
                    raise ValueError('instance IDs must be UINT')


def verify(record_path, inspect_exr=False):
    record = load(record_path)
    validate_record_shape(record)
    if not record['schema'] == 'quantiloom.dataset.export':
        raise ValueError(f"invalid record schema: {record['schema']!r}")
    if not (type(record['schema_version']) is int and record['schema_version'] in (1, 2)):
        raise ValueError(
            f"unsupported export schema version: {record['schema_version']!r}")
    if not record['state'] == record['capture_status'] == 'complete':
        raise ValueError('record is not complete')
    if not record['pairing_status'] in ('not_requested', 'pending', 'complete', 'failed'):
        raise ValueError(f"invalid pairing_status: {record['pairing_status']!r}")
    if not re.fullmatch('[0-9a-f]{64}', record['record_id']):
        raise ValueError(f"invalid record_id: {record['record_id']!r}")
    if not isinstance(record['provenance'], dict):
        raise ValueError('provenance must be an object')
    products = record['products']
    if not (isinstance(products, list) and products):
        raise ValueError('record has no products')
    names, ids = ({record_path.name.casefold()}, set())
    for artifact, is_product in [(record['replay'], False),
                                 *((p, True) for p in products)]:
        name = artifact['path']
        if not (isinstance(name, str) and name and (name not in ('.', '..'))):
            raise ValueError(f'invalid artifact path: {name!r}')
        relative = pathlib.PurePosixPath(name)
        segments_ok = all(
            p and p.casefold() not in ('.', '..', '.internal')
            and not p.casefold().endswith('.quantiloom-export.lock')
            and not p.endswith(('.', ' '))
            for p in name.split('/'))
        if not (not relative.is_absolute() and segments_ok):
            raise ValueError(f'unsafe artifact path: {name!r}')
        if not (not any((c in name for c in '\\:<>"|?*'))
                and (not name.endswith(('.', ' ')))):
            raise ValueError(f'artifact path has reserved characters: {name!r}')
        if not name.casefold() not in names:
            raise ValueError('duplicate artifact path')
        names.add(name.casefold())
        path = record_path.parent / name
        if not all((not p.is_symlink() for p in [path, *path.parents])):
            raise ValueError('symlink artifact path')
        if not (not path.is_symlink() and path.is_file()):
            raise ValueError(f'missing artifact: {name}')
        if not re.fullmatch('[0-9a-f]{64}', artifact['sha256']):
            raise ValueError(f"invalid sha256 for {name}: {artifact['sha256']!r}")
        if not digest(path) == artifact['sha256']:
            raise ValueError(f'hash mismatch: {name}')
        if 'size_bytes' in artifact:
            if not type(artifact['size_bytes']) is int:
                raise ValueError(
                    f"invalid size_bytes for {name}: {artifact['size_bytes']!r}")
            if not path.stat().st_size == artifact['size_bytes']:
                raise ValueError(f'size mismatch: {name}')
        if not is_product:
            continue
        product_id = artifact['product_id']
        if not (isinstance(product_id, str) and product_id and (product_id not in ids)):
            raise ValueError(f'invalid or duplicate product_id: {product_id!r}')
        ids.add(product_id)
        description = artifact['description']
        if not isinstance(description, dict):
            raise ValueError('product description must be an object')
        geometry = description.get('provenance', {}).get('geometry')
        if geometry:
            check_geometry(geometry)
            if not geometry['width'] == description['width']:
                raise ValueError('geometry width disagrees with description')
            if not geometry['height'] == description['height']:
                raise ValueError('geometry height disagrees with description')
        summary = None
        if path.suffix.lower() == '.png':
            summary = png_summary(path)
        if inspect_exr and path.suffix.lower() == '.exr':
            import OpenEXR
            import numpy as np
            with OpenEXR.File(str(path), separate_channels=True) as image:
                summary = dict(image.header())
                channels = image.channels()
                if 'channel_names' in description \
                        and set(channels) != set(description['channel_names']):
                    raise ValueError('EXR channel names disagree with description')
                if not len(channels) == description['channels']:
                    raise ValueError('EXR channel count disagrees with description')
                for channel in channels.values():
                    pixels = channel.pixels
                    if not pixels.shape[:2] == (description['height'], description['width']):
                        raise ValueError('EXR shape disagrees with description')
                    if not pixels.dtype in (np.dtype('float32'), np.dtype('uint32')):
                        raise ValueError(f'unsupported EXR dtype: {pixels.dtype}')
                    signal = description.get('signal')
                    if signal and pixels.dtype != np.dtype(signal['storage']):
                        raise ValueError('EXR storage type disagrees with signal')
                    if not np.isfinite(pixels).all():
                        raise ValueError('non-finite EXR pixels')
        if summary is not None:
            sidecar = '/'.join(['..'] * (len(pathlib.PurePosixPath(name).parts) - 1)
                               + [record_path.name])
            for key, expected in (('quantiloom_record_id', record['record_id']),
                                  ('quantiloom_product_id', product_id),
                                  ('quantiloom_sidecar', sidecar)):
                actual = summary[key]
                if isinstance(actual, bytes):
                    actual = actual.decode('utf-8')
                if not actual == expected:
                    raise ValueError(f'summary mismatch: {name}: {key}')
    return {'valid': True, 'checks': 'export_integrity',
            'product_count': len(products), 'reproducibility_verified': False,
            'exr_content_checked': inspect_exr}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('record', type=pathlib.Path)
    parser.add_argument('--inspect-exr', action='store_true')
    args = parser.parse_args()
    try:
        report = verify(args.record, args.inspect_exr)
    except Exception as exc:
        report = {'valid': False, 'checks': 'export_integrity',
                  'errors': [f'{type(exc).__name__}: {exc}']}
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report['valid'] else 1


if __name__ == '__main__':
    sys.exit(main())
