#!/usr/bin/env python3
"""Check UINT identity storage and moving offline cameras independently of the SDK.

Run after building the CLI and tests, with numpy and OpenEXR installed. These
are prerequisite checks, not acceptance of visibility annotations. Exit 0 pass,
1 failure, 2 missing binaries, 3 no GPU. Console output is ASCII.
"""
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import OpenEXR

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / 'build' / 'dataset_prerequisites'
CLI = ROOT / 'build/src/app/Release/Quantiloom.exe'
TESTS = ROOT / 'build/tests/Release/libquantiloom_tests.exe'


def child_path(path):
    value = str(path)
    match = re.match(r'^/mnt/([a-zA-Z])/(.*)$', value)
    return f'{match[1].upper()}:/{match[2]}' if match else value.replace('\\', '/')


def run(args, name, env=None):
    process = subprocess.run([str(x) for x in args], cwd=ROOT, env=env,
                             capture_output=True, encoding='utf-8', errors='replace', timeout=600)
    log = process.stdout + process.stderr
    (WORK / f'{name}.log').write_text(log, encoding='utf-8')
    if re.search(r'No Vulkan-compatible GPUs|Failed to create Vulkan instance|No suitable GPU', log):
        raise NoGpu()
    if process.returncode:
        raise ValueError(f'{name} failed; see {name}.log')


class NoGpu(Exception):
    pass


def check_uint():
    fixture = WORK / 'uint.exr'
    fixture.unlink(missing_ok=True)
    env = dict(os.environ, QUANTILOOM_UINT_EXR_FIXTURE=child_path(fixture))
    if sys.platform.startswith('linux'):
        env['WSLENV'] = ':'.join(filter(None, [env.get('WSLENV', ''), 'QUANTILOOM_UINT_EXR_FIXTURE']))
    run([TESTS, '--gtest_filter=ImageIOTest.UIntExr*:ExportSessionTest.*', '--gtest_brief=1'], 'integer_and_locks', env)
    with OpenEXR.File(str(fixture), separate_channels=True) as image:
        if set(image.channels()) != {'instance_id'}:
            raise ValueError('integer channel name mismatch')
        pixels = image.channels()['instance_id'].pixels
        expected = np.array([[0, 1, 16777217], [2147483649, 4294967295, 17]], dtype=np.uint32)
        if pixels.dtype != np.uint32 or not np.array_equal(pixels, expected):
            raise ValueError('UINT identity precision lost')
    print('PASS: independent UINT EXR read, including 2^24+1 and UINT_MAX')


def config(name, mode, pose, motion):
    # Root-derived paths: this fixture must work in another checkout.
    model = child_path(ROOT / 'assets/models/cornell_box/cornell_box.gltf')
    return f'''
[renderer]
resolution = [32, 24]
spp = 4
seed = 9812
output = "{child_path(WORK / (name + '.exr'))}"
[spectral]
mode = "{mode}"
wavelength_nm = 550.0
band = "VIS"
[scene]
gltf = "{model}"
world_units_to_meters = 0.001
[timeline]
start_s = 0.0
end_s = 2.0
time_s = 1.0
ticks_per_second = 1.0
[camera]
position = [{pose}, 273.0, -800.0]
look_at = [{pose}, 273.0, 280.0]
up = [0.0, 1.0, 0.0]
fov_y = 39.0
{motion}
[lighting]
sun_direction = [0.0, 1.0, 0.0]
sun_radiance = [0.0, 0.0, 0.0]
sky_radiance = [0.0, 0.0, 0.0]
[material_overrides."Light Panel"]
emissive_curve = "d65"
[material]
albedo = [0.5, 0.5, 0.5]
[quality]
fail_on_srgb_upsample = false
[sensor]
enabled = false
[dataset]
metadata = true
[hyperspectral]
wavelength_min_nm = 540.0
wavelength_max_nm = 560.0
wavelength_step_nm = 10.0
output_format = "exr_spectral"
use_gpu_reconstruction = false
save_intermediates = true
'''


def pixels(path):
    with OpenEXR.File(str(path), separate_channels=True) as image:
        return {name: channel.pixels.copy() for name, channel in image.channels().items()}


def check_camera(mode):
    motion = '''
[camera.motion]
interpolation = "linear"
extrapolate = "hold"
[[camera.motion.keys]]
t = 0.0
position = [278.0, 273.0, -800.0]
look_at = [278.0, 273.0, 280.0]
[[camera.motion.keys]]
t = 2.0
position = [478.0, 273.0, -800.0]
look_at = [478.0, 273.0, 280.0]
'''
    results = {}
    for kind, pose, track in [('moving', 278, motion), ('reference', 378, ''), ('start', 278, '')]:
        name = f'{mode}_{kind}'
        document = WORK / f'{name}.toml'
        document.write_text(config(name, mode, pose, track), encoding='utf-8')
        (WORK / f'{name}.exr').unlink(missing_ok=True)
        run([CLI, child_path(document)], name)
        results[kind] = pixels(WORK / f'{name}.exr')
        sys.path.insert(0, str(ROOT / 'scripts'))
        import dataset_verify
        report = dataset_verify.verify(WORK / f'{name}.metadata.json', inspect_exr=True)
        if not report['valid']:
            raise ValueError(f'{name}: independent integrity verification failed')
    reference = results['reference']
    if set(results['moving']) != set(reference):
        raise ValueError(f'{mode}: product channels differ')
    if not all(np.array_equal(results['moving'][c], reference[c]) for c in reference):
        raise ValueError(f'{mode}: moving pose differs from authored reference at t=1')
    if all(np.array_equal(results['start'][c], reference[c]) for c in reference):
        raise ValueError(f'{mode}: fixture did not distinguish the two poses')
    record = json.loads((WORK / f'{mode}_moving.metadata.json').read_text(encoding='utf-8'))
    geometry = record['provenance']['geometry']
    matrix = geometry['camera_to_world']
    if geometry['reference_time_s'] != 1 or abs(matrix[3] - 378) > 1e-5:
        raise ValueError(f'{mode}: frozen geometry disagrees with actual pose')
    print(f'PASS: {mode} moving pixels equal static reference, actual pose frozen')


def main():
    if not CLI.is_file() or not TESTS.is_file():
        print('Missing CLI or test binary')
        return 2
    WORK.mkdir(parents=True, exist_ok=True)
    try:
        check_uint()
        check_camera('single')
        check_camera('multispectral')
    except NoGpu:
        print('No usable GPU; camera checks not performed')
        return 3
    except Exception as error:
        print(f'FAIL: {type(error).__name__}: {error}')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
