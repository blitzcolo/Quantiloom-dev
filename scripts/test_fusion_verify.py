"""Binary compatibility and malformed-column checks without an SDK."""
import copy
import pathlib
import struct
import tempfile
import unittest
import fusion_verify
from fusion_verify import (kFusionHeaderBytes, kFusionHeaderWords,
                           kFusionMaxSlotsPerRay, kFusionRayIdentityOffset,
                           kFusionRayRecordBytes, kFusionTerminalBit,
                           kFusionVertexRecordBytes)


def pack_column(data, name_map, name, fmt, *values):
    struct.pack_into(fmt, data, name_map[name]['offset_bytes'], *values)


class FusionColumnsTest(unittest.TestCase):
    def fixture(self):
        """One stored ray, one valid terminal vertex, all ten columns."""
        slots = kFusionMaxSlotsPerRay
        names = [('components', 'float32', 1, 4), ('radiance', 'float32', 1, 1),
                 ('ray_identity', 'uint32', 1, 4), ('native_pixel', 'float32', 1, 2),
                 ('position_wavelength', 'float32', slots, 4),
                 ('normal_distance', 'float32', slots, 4),
                 ('vertex_identity', 'uint32', slots, 4),
                 ('outgoing_coefficient', 'float32', slots, 4),
                 ('medium_segment', 'float32', slots, 4),
                 ('vertex_radiance_terms', 'float32', slots, 4)]
        data = bytearray()
        columns = []
        for name, kind, rows, width in names:
            offset = len(data)
            data.extend(bytes(rows * width * 4))
            columns.append(dict(name=name, type=kind, rows=rows, width=width,
                                offset_bytes=offset, size_bytes=rows * width * 4))
        struct.pack_into('<4f', data, 0, 1, 2, 3, 4)
        struct.pack_into('<f', data, 16, 10)
        struct.pack_into('<4I', data, 20, 0, 0, 0, kFusionTerminalBit)
        by_name = {c['name']: c for c in columns}
        pack_column(data, by_name, 'medium_segment', '<4f', 1, 1, 0, 1)
        pack_column(data, by_name, 'vertex_radiance_terms', '<4f', 10, 0, 0, 0)
        description = dict(schema_version=2, byte_order='little_endian', stored_rays=1,
                           slots_per_ray=slots, ray_stride=1, native_width=1,
                           native_height=1, spp=1, diagnostic_flags=0,
                           columns=columns)
        return data, description

    def test_roundtrip_and_trailing_bytes(self):
        data, d = self.fixture()
        self.assertEqual(fusion_verify.path_columns(data, d)['components'][0],
                         (1, 2, 3, 4))
        with self.assertRaises(ValueError):
            fusion_verify.path_columns(data + b'x', d)

    def test_overlap_wrong_type_and_missing_column(self):
        data, d = self.fixture()
        mutations = (lambda x: x['columns'][1].update(offset_bytes=0),
                     lambda x: x['columns'][0].update(type='uint32'),
                     lambda x: x['columns'].pop(),
                     lambda x: x.update(diagnostic_flags=16))
        for mutate in mutations:
            bad = copy.deepcopy(d)
            mutate(bad)
            with self.assertRaises(ValueError):
                fusion_verify.path_columns(data, bad)

    def test_reconstruction_is_not_forced(self):
        data, d = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / 'paths.bin'
            path.write_bytes(data)
            fusion_verify.check_paths(path, d)
            struct.pack_into('<f', data, 16, 11)
            path.write_bytes(data)
            with self.assertRaises(ValueError):
                fusion_verify.check_paths(path, d)

    def test_v1_layout_still_reads(self):
        slots = kFusionMaxSlotsPerRay
        header = [1, 1, 1, slots, 1, 1, 1, 0] + [0] * (kFusionHeaderWords - 8)
        data = bytearray(struct.pack(f'<{kFusionHeaderWords}I', *header)
                         + bytes(kFusionRayRecordBytes + slots * kFusionVertexRecordBytes))
        struct.pack_into('<4f', data, kFusionHeaderBytes, 1, 2, 3, 6)
        struct.pack_into('<4I', data, kFusionHeaderBytes + kFusionRayIdentityOffset,
                         0, 0, 0, kFusionTerminalBit)
        d = dict(schema_version=1, ray_stride=1, stored_rays=1, slots_per_ray=slots,
                 native_width=1, native_height=1, spp=1, diagnostic_flags=0)
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / 'v1.bin'
            path.write_bytes(data)
            fusion_verify.check_paths(path, d)


if __name__ == '__main__':
    unittest.main()
