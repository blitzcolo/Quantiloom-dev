"""Independent CPU geometric optics checks against authored triangles and temperatures.

Snell/Fresnel and Beer-Lambert equations follow PBRT 4e Dielectric BSDF and
Transmittance. Recorded branch decisions select a path; geometry, directions,
interface weights and radiance are recomputed independently from source data.
"""
import base64
import json
import math
import re
import sys
from pathlib import Path
import tomllib
import numpy as np
import fusion_verify
from fusion_verify import kFusionTerminalBit, kFusionTerminalDepthMask

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'physics-audit'))
import harness

# Per-chunk cap on independently retraced paths; keeps the check bounded on
# dense captures without weakening the per-path assertions.
MAX_PATHS_PER_CHUNK = 128


def resolve_referenced_path(base, value):
    """Resolve a path written into a generated TOML for the Windows exe.

    Absolute POSIX paths are used as-is; an absolute drive-letter path
    (H:/..., what win_path writes under WSL) maps back to /mnt/<drive>/... so
    this host-side check can read the same file; relative paths join `base`.
    """
    path = Path(value)
    if path.is_absolute():
        return path
    match = re.match(r'^([a-zA-Z]):[/\\](.*)$', value)
    if match:
        return Path('/mnt') / match[1].lower() / match[2]
    return base / path


def triangles(path):
    doc = json.loads(path.read_text())
    buffers = [
        base64.b64decode(b['uri'].split(',', 1)[1])
        if b['uri'].startswith('data:')
        else (path.parent / b['uri']).read_bytes()
        for b in doc['buffers']
    ]

    def accessor(index):
        a = doc['accessors'][index]
        v = doc['bufferViews'][a['bufferView']]
        width = {'SCALAR': 1, 'VEC3': 3}[a['type']]
        dtype = {5126: '<f4', 5125: '<u4', 5123: '<u2', 5121: 'u1'}[a['componentType']]
        stride = v.get('byteStride', np.dtype(dtype).itemsize * width)
        offset = v.get('byteOffset', 0) + a.get('byteOffset', 0)
        return np.ndarray(
            (a['count'], width), dtype, buffer=buffers[v['buffer']],
            offset=offset, strides=(stride, np.dtype(dtype).itemsize)).copy()

    points, ids, materials = [], [], []

    def visit(index, parent):
        node = doc['nodes'][index]
        if 'matrix' in node:
            local = np.array(node['matrix']).reshape(4, 4).T
        else:
            local = np.eye(4)
            local[:3, :3] = np.diag(node.get('scale', [1, 1, 1]))
            local[:3, 3] = node.get('translation', [0, 0, 0])
            if 'rotation' in node:
                raise ValueError('reference fixture rotation requires a matrix')
        world = parent @ local
        if 'mesh' in node:
            for p in doc['meshes'][node['mesh']]['primitives']:
                positions = accessor(p['attributes']['POSITION'])
                positions = (np.c_[positions, np.ones(len(positions))] @ world.T)[:, :3]
                indices = accessor(p['indices']).reshape(-1, 3)
                points.extend(positions[indices])
                ids.extend([index + 1] * len(indices))
                materials.extend(
                    [doc['materials'][p['material']]['name']] * len(indices))
        for child in node.get('children', []):
            visit(child, world)

    for node in doc['scenes'][doc.get('scene', 0)]['nodes']:
        visit(node, np.eye(4))
    return np.array(points), np.array(ids), materials


def check_package(manifest_path, scene_path):
    manifest_path = Path(manifest_path)
    scene_path = Path(scene_path)
    scene = tomllib.loads(scene_path.read_text())
    tri, ids, names = triangles(
        resolve_referenced_path(scene_path.parent, scene['scene']['gltf']))
    a = tri[:, 0]
    e1 = tri[:, 1] - a
    e2 = tri[:, 2] - a
    normals = np.cross(e1, e2)
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    overrides = scene['material_overrides']
    manifest = json.loads(manifest_path.read_text())

    def ior(material, wavelength):
        nd = float(material.get('ior', 1.5))
        reciprocal_abbe = float(material.get('dispersion', 0))
        if reciprocal_abbe <= 0:
            return nd
        # Independently solve A+B/lambda^2 from n_d and the F/C Abbe difference.
        B = (nd - 1) * reciprocal_abbe / (1 / 486.13**2 - 1 / 656.27**2)
        A = nd - B / 587.56**2
        return max(1.0, A + B / wavelength**2)

    def intersect_all(origin, direction, u_eps=0, t_min=1e-8):
        """Moller-Trumbore against every triangle; returns (t, u, v, det)."""
        h = np.cross(np.broadcast_to(direction, e2.shape), e2)
        det = np.einsum('ij,ij->i', e1, h)
        inv = np.divide(1, det, out=np.zeros_like(det), where=np.abs(det) > 1e-12)
        s = origin - a
        u = np.einsum('ij,ij->i', s, h) * inv
        q = np.cross(s, e1)
        v = q @ direction * inv
        t = np.einsum('ij,ij->i', e2, q) * inv
        valid = ((np.abs(det) > 1e-12) & (u >= -u_eps) & (v >= -u_eps)
                 & (u + v <= 1 + u_eps) & (t > t_min))
        return t, u, v, valid

    def initial_media(origin):
        # Which solid media contain the ray origin, innermost last. Derived
        # from even-odd crossing counts along a probe direction that hits no
        # authored edge.
        direction = np.array([1.0, 0.37, 0.79])
        direction /= np.linalg.norm(direction)
        t, _, _, valid = intersect_all(origin, direction)
        inside = []
        for node in np.unique(ids):
            name = names[int(np.flatnonzero(ids == node)[0])]
            if overrides.get(name, {}).get('fusion_transport') != 'solid':
                continue
            crossings = np.unique(np.round(t[valid & (ids == node)], 9))
            if len(crossings) % 2:
                inside.append((crossings[0], (int(node), name)))
        return [entry for distance, entry in sorted(inside, reverse=True)]

    def hit(origin, direction):
        t, _, _, valid = intersect_all(origin, direction, u_eps=1e-9, t_min=1e-6)
        distances = np.where(valid, t, np.inf)
        index = int(np.argmin(distances))
        return (index, float(distances[index])) if math.isfinite(distances[index]) else None

    checked = 0
    max_error = 0
    for truth in manifest['ground_truth']:
        origin = np.array(truth['geometry']['camera_to_world']).reshape(4, 4)[:3, 3]
        for chunk in truth['path_chunks']:
            chunk_checked = 0
            desc = chunk['description']
            cols = fusion_verify.path_columns(
                (manifest_path.parent / chunk['path']).read_bytes(), desc)
            slots = desc['slots_per_ray']
            wavelength = desc['wavelength_nm']
            for row, identity in enumerate(cols['ray_identity']):
                if chunk_checked >= MAX_PATHS_PER_CHUNK:
                    break
                if identity[2] or not identity[3] & kFusionTerminalBit:
                    continue
                depth = identity[3] & kFusionTerminalDepthMask
                first = np.array(cols['position_wavelength'][row * slots][:3])
                d = first - origin
                d /= np.linalg.norm(d)
                o = origin.copy()
                media = initial_media(origin)
                operations = []
                terminal = None
                for step in range(depth + 1):
                    intersection = hit(o, d)
                    record = row * slots + step
                    kind, node, primitive, route = cols['vertex_identity'][record]
                    if intersection is None:
                        terminal = harness.planck_blackbody(260, wavelength)
                        break
                    index, distance = intersection
                    point = o + d * distance
                    recorded = np.array(cols['position_wavelength'][record][:3])
                    if int(ids[index]) != node or np.linalg.norm(point - recorded) > 2e-4:
                        raise ValueError(
                            'independent optical geometry disagrees with path record')
                    segment_trans = 1
                    segment_emission = 0
                    if media:
                        medium = overrides[media[-1][1]]
                        n = ior(medium, wavelength)
                        sigma = float(medium.get('fusion_absorption_m_inv', 0))
                        segment_trans = math.exp(
                            -sigma * distance * truth['geometry']['world_units_to_meters'])
                        segment_emission = ((1 - segment_trans) * n * n
                                            * harness.planck_blackbody(
                                                medium.get('ir_temperature_k', 300),
                                                wavelength))
                    material = overrides.get(names[index], {})
                    if kind == 3:
                        terminal = harness.planck_blackbody(
                            material.get('ir_temperature_k', 300), wavelength)
                        operations.append((1, 0, segment_trans, segment_emission))
                        break
                    normal = normals[index].copy()
                    back = np.dot(normal, d) > 0
                    if back:
                        normal = -normal
                    out = np.array(cols['outgoing_coefficient'][record][:3])
                    reflected = np.dot(out, normal) > 0
                    n1 = ior(overrides[media[-1][1]], wavelength) if media else 1.0
                    if kind == 1:
                        rho = float(material['fusion_sheet_reflectance'])
                        tau = float(material['fusion_sheet_transmittance'])
                        n2 = n1
                        self_emission = ((1 - rho - tau) * n1 * n1
                                         * harness.planck_blackbody(
                                             material.get('ir_temperature_k', 300),
                                             wavelength))
                        next_d = d - 2 * np.dot(d, normal) * normal if reflected else d
                        weight = rho + tau
                    else:
                        if not back:
                            n2 = ior(material, wavelength)
                        else:
                            n2 = (ior(overrides[media[-2][1]], wavelength)
                                  if len(media) > 1 else 1.0)
                        cosine = -float(np.dot(d, normal))
                        eta = n1 / n2
                        sin2 = eta * eta * (1 - cosine * cosine)
                        if sin2 >= 1:
                            rho = 1
                            refracted = None
                        else:
                            ct = math.sqrt(max(0, 1 - sin2))
                            rs = (n1 * cosine - n2 * ct) / (n1 * cosine + n2 * ct)
                            rp = (n2 * cosine - n1 * ct) / (n2 * cosine + n1 * ct)
                            rho = (rs * rs + rp * rp) / 2
                            refracted = eta * d + (eta * cosine - ct) * normal
                        tau = 1 - rho
                        self_emission = 0
                        next_d = (d - 2 * np.dot(d, normal) * normal
                                  if reflected else refracted)
                        if next_d is None:
                            raise ValueError(
                                'transmitted path violates total internal reflection')
                        weight = 1 if reflected else n1 * n1 / (n2 * n2)
                        if not reflected:
                            if back:
                                if not media or media[-1][0] != node:
                                    raise ValueError(
                                        'independent nested medium order mismatch')
                                media.pop()
                            else:
                                media.append((node, names[index]))
                    coefficient = ((rho if reflected else tau)
                                   * (n1 * n1 / (n2 * n2)
                                      if kind == 2 and not reflected else 1))
                    if (np.linalg.norm(next_d - out) > 1e-4
                            or abs(coefficient
                                   - cols['outgoing_coefficient'][record][3]) > 1e-4):
                        raise ValueError('independent Snell/Fresnel check failed')
                    operations.append((weight, self_emission, segment_trans,
                                       segment_emission))
                    o = point
                    d = next_d
                if terminal is None:
                    continue
                value = terminal
                for weight, emission, trans, bulk in reversed(operations):
                    value = (value * weight + emission) * trans + bulk
                measured = cols['radiance'][row][0]
                error = abs(value - measured) / max(abs(value), 1e-12)
                max_error = max(max_error, error)
                if error > 0.01:
                    raise ValueError(
                        'independent optical radiance differs by more than 1 percent')
                checked += 1
                chunk_checked += 1
    if checked < 32:
        raise ValueError('too few independent optical paths checked')
    print('Independent optics passed: %d paths; max radiance error %.4f percent'
          % (checked, 100 * max_error))
