#!/usr/bin/env python3
"""Fusion contract and native-camera geometry gate; all output is ASCII.

Runs the standalone fusion_verify unit checks, then drives `Quantiloom.exe
fusion-export` over the shipped fixtures and re-derives the quantities the GPU
measured:

  projection   -- world_position ground truth reprojected through each native
                  lens model must land on its own pixel to 0.01 px
  transmission -- a homogeneous slab's recorded radiance against the closed
                  form (Planck r/t/a), plus verified optical correspondences
  portability  -- a copied export package must still verify standalone
  optics       -- an independent CPU Snell/Fresnel walk over the recorded paths

Exit 0 pass, 1 failure, 2 missing binaries, 3 no usable GPU.
"""
import json
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import fusion_verify

REPO = pathlib.Path(__file__).resolve().parents[2]
CLI = REPO / 'build/src/app/Release/Quantiloom.exe'
UNIT = REPO / 'scripts/test_fusion_verify.py'
FIXTURES = REPO / 'docs/dataset/fusion'

# A failure log line that means "no device to measure on" rather than a wrong
# answer. Same vocabulary as check_camera_dynamic.py, plus the ray-tracing
# variant SelectPhysicalDevice reports when devices exist but none qualify.
NO_GPU = re.compile(
    r'No Vulkan-compatible GPUs|Failed to create Vulkan instance|'
    r'No suitable|No GPU with Ray Tracing support|CUDA_ERROR|device lost',
    re.IGNORECASE,
)


def win_path(path):
    """Convert /mnt/<drive>/... to <drive>:/... for the Windows child .exe.

    WSL interop translates the child's working directory but not its argv, and
    the same applies to paths embedded in the TOML files the exe reads. On
    native Windows the string is already a Windows path and passes through.
    """
    text = str(path)
    match = re.match(r'^/mnt/([a-zA-Z])/(.*)$', text)
    if match:
        return f'{match[1].upper()}:/{match[2]}'
    return text


def replace_once(text, old, new):
    """Substitute a unique marker. A fixture edit must never silently no-op:
    if `old` is absent or ambiguous the check is wrong, not the render."""
    if text.count(old) != 1:
        raise ValueError(f'fixture marker {old!r} occurs {text.count(old)} times')
    return text.replace(old, new)


def replace_all(text, old, new):
    """Substitute every occurrence of a marker that must occur at least once."""
    if old not in text:
        raise ValueError(f'fixture marker {old!r} not found')
    return text.replace(old, new)


def run_render(job_path, name):
    """Run fusion-export and return the completed process.

    Failure text lands on stderr while logging lands on stdout, so the no-GPU
    decision looks at both; anything else that fails is a wrong render.
    """
    render = subprocess.run(
        [str(CLI), 'fusion-export', win_path(job_path)],
        cwd=REPO, capture_output=True, text=True, errors='replace')
    if render.returncode:
        log = (render.stdout or '') + (render.stderr or '')
        if NO_GPU.search(log):
            raise NoGpu(f'{name}: no usable GPU, nothing measured')
        raise ValueError(f'{name} render failed: {render.stderr[-1000:]}')
    return render


class NoGpu(Exception):
    pass


def raster(products, package, product_id):
    """Read one product's channels as a (height, width, channels) array."""
    import OpenEXR
    import numpy as np
    product = products[product_id]
    with OpenEXR.File(str(package / product['path']), separate_channels=True) as image:
        names = product['description']['channel_names']
        return np.stack([image.channels()[name].pixels for name in names], axis=-1)


def distort(distortion, x, y):
    """Apply the authored distortion model to unit-plane camera coordinates."""
    import numpy as np
    if distortion['model'] == 'brown_conrady':
        k1, k2, p1, p2, k3 = distortion['coefficients']
        r = x * x + y * y
        a = 1 + k1 * r + k2 * r * r + k3 * r * r * r
        return (x * a + 2 * p1 * x * y + p2 * (r + 2 * x * x),
                y * a + p1 * (r + 2 * y * y) + 2 * p2 * x * y)
    if distortion['model'] == 'fisheye':
        r = np.sqrt(x * x + y * y)
        theta = np.arctan(r)
        td = theta * (1 + sum(distortion['coefficients'][i] * theta ** (2 * i + 2)
                              for i in range(4)))
        scale = np.divide(td, r, out=np.ones_like(r), where=r > 0)
        return x * scale, y * scale
    return x, y


def reprojection_error(truth, products, package):
    """Reproject recorded world positions through the truth's own lens model
    and return (largest pixel error, opaque pixel count), or None when the
    truth has no opaque pixels -- the caller decides whether that is a wrong
    fixture or a camera that legitimately saw nothing."""
    import numpy as np
    world = raster(products, package,
                   next(p['product_id'] for p in truth['products']
                        if p['product_id'].endswith('/world_position')))
    mask = raster(products, package,
                  next(p['product_id'] for p in truth['products']
                       if p['product_id'].endswith('/geometry_validity')))[..., 0] == 1
    if not mask.any():
        return None
    points = world[mask].astype(np.float64)
    transform = np.array(truth['geometry']['world_to_camera']).reshape(4, 4)
    camera = points @ transform[:3, :3].T + transform[:3, 3]
    x, y = distort(truth['geometry']['distortion'], camera[:, 0] / camera[:, 2],
                   camera[:, 1] / camera[:, 2])
    yy, xx = np.nonzero(mask)
    k = truth['geometry']['intrinsics']
    error = np.maximum(np.abs(k[0] * x + k[2] - (xx + .5)),
                       np.abs(k[4] * y + k[5] - (yy + .5)))
    return float(error.max()), int(mask.sum())


def check_baseline():
    """The example rig: reprojection through two native cameras, verified
    correspondences, and a manifest the standalone verifier accepts."""
    run_render(FIXTURES / 'example.toml', 'example')
    package = REPO / 'build/fusion-example'
    manifest_path = package / 'plate.manifest.json'
    report = fusion_verify.verify(manifest_path, True)
    manifest = json.loads(manifest_path.read_text())
    record = json.loads((package / manifest['record_path']).read_text())
    products = {p['product_id']: p for p in record['products']}
    checked = 0
    for truth in manifest['ground_truth']:
        result = reprojection_error(truth, products, package)
        if result is None:
            raise ValueError('geometry fixture has no opaque hits')
        error, pixels = result
        if error > .01:
            raise ValueError('GPU projection exceeds 0.01 pixel tolerance')
        checked += pixels
    for pair in manifest['pairs']:
        mask = raster(products, package, pair['validity_product'])[..., 0]
        if not (mask == 1).any():
            raise ValueError('pair fixture has no verified correspondences')
    print('Fusion gate passed: %d cameras, %d pairs, %d checked opaque pixels'
          % (report['camera_count'], report['pair_count'], checked))


def variant_job(work, name, model, coefficients):
    """Rewrite the example job for one lens model into a working directory."""
    original = (FIXTURES / 'example.toml').read_text()
    doc = replace_once(original, 'scene_config = "../example/scene.toml"',
                       'scene_config = ' + json.dumps(
                           win_path(FIXTURES.parent / 'example/scene.toml')))
    doc = replace_once(doc, 'output_directory = "../../../build/fusion-example"',
                       'output_directory = ' + json.dumps(win_path(work / name)))
    doc = replace_once(doc, 'seed = 42', 'rectify = true\nseed = 42')
    doc = replace_once(doc, 'model = "brown_conrady"',
                       'model = ' + json.dumps(model))
    doc = replace_once(doc,
                       'coefficients = [-0.08, 0.01, 0.001, -0.002, 0.0]',
                       'coefficients = ' + coefficients)
    job = work / (name + '.toml')
    job.write_text(doc)
    return job


def check_lens_variants():
    """Exercise actual GPU rays for nonzero fisheye and native/derived grids:
    world positions must reproject through each distortion model, a rectified
    raster variant must be present, and the package must verify after being
    moved wholesale to another directory."""
    import numpy as np
    work = REPO / 'build/fusion-lens-check'
    work.mkdir(exist_ok=True)
    for name, model, coefficients in [
            ('zero', 'brown_conrady', '[0,0,0,0,0]'),
            ('pincushion', 'brown_conrady', '[0.08,0.01,-0.001,0.002,0.001]'),
            ('fisheye', 'fisheye', '[0.015,-0.002,0.0001,-0.00001]')]:
        job = variant_job(work, name, model, coefficients)
        run_render(job, f'lens variant {name}')
        package = work / name
        manifest_path = package / 'plate.manifest.json'
        fusion_verify.verify(manifest_path, True)
        manifest = json.loads(manifest_path.read_text())
        record = json.loads((package / manifest['record_path']).read_text())
        products = {p['product_id']: p for p in record['products']}
        checked = 0
        for truth in manifest['ground_truth']:
            result = reprojection_error(truth, products, package)
            if result is None:
                continue
            error, pixels = result
            if error > .01:
                raise ValueError('lens variant GPU reprojection exceeds 0.01 pixel')
            checked += pixels
        if checked == 0 or not any(t.get('raster_variant') == 'rectified'
                                   for t in manifest['ground_truth']):
            raise ValueError('derived lens geometry missing')
        for observation in manifest['observations']:
            sensor = tomllib.loads(observation['sensor_config_toml'])['sensor']
            psf = sensor['optics'].get('known_psf_path', '')
            if psf and (pathlib.PurePosixPath(psf).is_absolute()
                        or not (package / psf).is_file()):
                raise ValueError('PSF is not portable')
        with tempfile.TemporaryDirectory() as temp:
            moved = pathlib.Path(temp) / 'portable'
            shutil.copytree(package, moved)
            fusion_verify.verify(moved / manifest_path.name, True)
        print('Lens variant passed: %s, %d native/derived GPU pixels; '
              'portable package verified' % (name, checked))


def check_transmission():
    """Homogeneous and curved media: recorded slab radiance against the
    closed-form Planck r/t/a, at least ten verified correspondences per
    fixture, then the independent optics reference over the path records."""
    sys.path.insert(0, str(REPO / 'scripts/physics-audit'))
    import harness
    import struct

    run_render(FIXTURES / 'transmission_job.toml', 'transmission')
    out = REPO / 'build/fusion-transmission'
    manifest_path = out / 'glass.manifest.json'
    manifest = json.loads(manifest_path.read_text())
    fusion_verify.verify(manifest_path, True)
    reflectance, transmittance, absorptance = \
        harness.homogeneous_slab_coefficients(1.5, 10, .2)
    max_error = 0.0
    for truth in manifest['ground_truth']:
        for chunk in truth['path_chunks']:
            data = (out / chunk['path']).read_bytes()
            description = chunk['description']
            count = description['stored_rays']
            columns = (fusion_verify.path_columns(data, description)
                       if description['schema_version'] == 2 else None)
            measured = sum(row[0] for row in columns['radiance']) / count \
                if columns \
                else sum(struct.unpack_from('<4f', data, 128 + i * 32)[3]
                         for i in range(count)) / count
            wavelength = description['wavelength_nm']
            expected = (transmittance * harness.planck_blackbody(350, wavelength)
                        + absorptance * harness.planck_blackbody(300, wavelength)
                        + reflectance * harness.planck_blackbody(260, wavelength))
            error = abs(measured / expected - 1)
            max_error = max(max_error, error)
            if error > .01:
                raise ValueError('slab radiance error exceeds 1 percent')
            slots = description['slots_per_ray']
            has_target = any(
                columns['vertex_identity'][i * slots + 2][1] == 2
                for i in range(count)) if columns else any(
                    struct.unpack_from('<4I', data,
                                       128 + count * 32 + (i * 9 + 2) * 80 + 32)[1] == 2
                    for i in range(count))
            if not has_target:
                raise ValueError('no real target behind glass in the trace records')
    optical_path = out / 'ground_truth/pairs/offset_to_reference_optical.json'
    optical = json.loads(optical_path.read_text())
    verified = sum(bool(row['matches']) for row in optical['rows'])
    if verified < 10:
        raise ValueError('refractive correspondence fixture has too few verified paths')
    print('Transmission gate passed: slab error %.4f percent; %d verified source paths'
          % (100 * max_error, verified))

    from fusion_optics_reference import check_package
    for case in ('curved', 'thin', 'nested'):
        run_render(FIXTURES / f'{case}_transmission_job.toml', f'{case} transmission')
        sample = REPO / 'build' / f'fusion-{case}' / f'{case}.manifest.json'
        fusion_verify.verify(sample, True)
        mapping_path = sample.parent / 'ground_truth/pairs/offset_to_reference_optical.json'
        mapping = json.loads(mapping_path.read_text())
        matched = sum(bool(r['matches']) for r in mapping['rows'])
        if matched < 10:
            raise ValueError(f'{case} has too few verified optical paths')
        print('%s optical gate passed: %d verified source paths' % (case, matched))
        check_package(sample, FIXTURES / f'{case}_transmission_scene.toml')

    # Independent dispersion and camera-inside-medium checks on a broader IR band.
    work = REPO / 'build/fusion-optical-variants'
    work.mkdir(exist_ok=True)
    scene_text = (FIXTURES / 'transmission_scene.toml').read_text()
    job_text = (FIXTURES / 'transmission_job.toml').read_text()
    for name in ('dispersion', 'inside'):
        source = replace_once(scene_text, 'gltf="transmission.gltf"',
                              'gltf=' + json.dumps(
                                  win_path(FIXTURES / 'transmission.gltf')))
        if name == 'dispersion':
            source = replace_once(source, 'ior=1.5', 'ior=1.5\ndispersion=0.05')
        scene_path = work / (name + '_scene.toml')
        scene_path.write_text(source)
        doc = replace_once(job_text, 'scene_config="transmission_scene.toml"',
                           'scene_config=' + json.dumps(win_path(scene_path)))
        doc = replace_once(doc, 'output_directory="../../../build/fusion-transmission"',
                           'output_directory=' + json.dumps(win_path(work / name)))
        doc = replace_once(doc, 'sample_id="glass"',
                           'sample_id=' + json.dumps(name))
        if name == 'inside':
            # Rig camera z: 2 -> 0.6 starts the ray inside the slab.
            doc = replace_all(doc, '0,0,-1,2,', '0,0,-1,0.6,')
        else:
            doc = replace_all(doc, '9999.0', '8000.0')
            doc = replace_all(doc, '10001.0', '12000.0')
            doc = replace_all(doc, 'wavelength_samples=1', 'wavelength_samples=3')
        job_path = work / (name + '.toml')
        job_path.write_text(doc)
        run_render(job_path, name)
        sample = work / name / (name + '.manifest.json')
        fusion_verify.verify(sample, True)
        check_package(sample, scene_path)


def run():
    unit = subprocess.run([sys.executable, str(UNIT)],
                          cwd=REPO, capture_output=True, text=True)
    if unit.returncode:
        raise ValueError('fusion binary contract tests failed: ' + unit.stderr[-1000:])
    check_baseline()
    check_lens_variants()
    check_transmission()
    from fusion_rectification_reference import check
    try:
        check(REPO, CLI)
    except ValueError as error:
        # The callee's render carries the child's stderr in its message; a
        # no-device failure there means the same skip as our own renders.
        if NO_GPU.search(str(error)):
            raise NoGpu(str(error))
        raise
    return 0


if __name__ == '__main__':
    if not CLI.is_file() or not UNIT.is_file():
        print('Fusion gate failed: missing CLI or contract test script')
        sys.exit(2)
    try:
        sys.exit(run())
    except NoGpu as error:
        print(f'SKIP fusion gate: {error}')
        sys.exit(3)
    except Exception as error:
        print('Fusion gate failed: ' + str(error))
        sys.exit(1)
