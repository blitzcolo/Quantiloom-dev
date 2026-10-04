"""Reject correspondence into invalid derived texels in a fisheye boundary case."""
import json
import subprocess
import OpenEXR
import numpy as np


def check(root, cli):
    job = root / 'docs/dataset/fusion/rectification_edge_job.toml'
    render = subprocess.run([str(cli), 'fusion-export', str(job)], cwd=root,
                            capture_output=True, text=True, errors='replace')
    if render.returncode:
        raise ValueError('rectification boundary render failed: ' + render.stderr[-1000:])
    package = root / 'build/fusion-rectification-edge'
    manifest = json.loads((package / 'edge.manifest.json').read_text())
    record = json.loads((package / manifest['record_path']).read_text())
    products = {p['product_id']: p for p in record['products']}

    def read(identity):
        p = products[identity]
        with OpenEXR.File(str(package / p['path']), separate_channels=True) as f:
            return np.stack([f.channels()[n].pixels for n in p['description']['channel_names']], axis=-1)

    pair = next(p for p in manifest['pairs'] if p['mapping'].endswith('rectified_pixel_centres'))
    valid = read(pair['validity_product'])[..., 0]
    coords = read(pair['coordinates_product'])
    target = read('edge/reference/rectification_valid')[..., 0]
    if not np.any(valid == 1) or not np.any(target == 0):
        raise ValueError('boundary fixture did not exercise valid and invalid regions')
    for y, x in zip(*np.where(valid == 1)):
        tx, ty = coords[y, x]
        ix, iy = int(round(float(tx)-.5)), int(round(float(ty)-.5))
        if abs(tx-ix-.5) > .001 or abs(ty-iy-.5) > .001:
            raise ValueError('co-located fixture lost centre alignment')
        if target[iy, ix] == 0:
            raise ValueError('valid correspondence points into invalid rectification support')
    print('Rectification boundary correspondence gate passed')
