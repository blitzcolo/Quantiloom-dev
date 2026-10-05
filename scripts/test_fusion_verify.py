"""Binary compatibility and malformed-column checks without an SDK."""
import copy
import pathlib
import struct
import tempfile
import unittest
import fusion_verify


class FusionColumnsTest(unittest.TestCase):
    def fixture(self):
        names=[('components','float32',1,4),('radiance','float32',1,1),('ray_identity','uint32',1,4),
               ('native_pixel','float32',1,2),('position_wavelength','float32',9,4),
               ('normal_distance','float32',9,4),('vertex_identity','uint32',9,4),
               ('outgoing_coefficient','float32',9,4),('medium_segment','float32',9,4),('vertex_radiance_terms','float32',9,4)]
        data=bytearray();columns=[]
        for name,kind,rows,width in names:
            offset=len(data);data.extend(bytes(rows*width*4))
            columns.append(dict(name=name,type=kind,rows=rows,width=width,offset_bytes=offset,size_bytes=rows*width*4))
        struct.pack_into('<4f',data,0,1,2,3,4);struct.pack_into('<f',data,16,10)
        struct.pack_into('<4I',data,20,0,0,0,0x80000000)
        by_name={c['name']:c for c in columns}
        struct.pack_into('<4f',data,by_name['medium_segment']['offset_bytes'],1,1,0,1)
        struct.pack_into('<4f',data,by_name['vertex_radiance_terms']['offset_bytes'],10,0,0,0)
        description=dict(schema_version=2,byte_order='little_endian',stored_rays=1,slots_per_ray=9,
                         ray_stride=1,native_width=1,native_height=1,spp=1,diagnostic_flags=0,columns=columns)
        return data,description

    def test_roundtrip_and_trailing_bytes(self):
        data,d=self.fixture()
        self.assertEqual(fusion_verify.path_columns(data,d)['components'][0],(1,2,3,4))
        with self.assertRaises(ValueError):fusion_verify.path_columns(data+b'x',d)

    def test_overlap_wrong_type_and_missing_column(self):
        data,d=self.fixture()
        for mutate in (lambda x:x['columns'][1].update(offset_bytes=0),
                       lambda x:x['columns'][0].update(type='uint32'),
                       lambda x:x['columns'].pop(),lambda x:x.update(diagnostic_flags=16)):
            bad=copy.deepcopy(d);mutate(bad)
            with self.assertRaises(ValueError):fusion_verify.path_columns(data,bad)

    def test_reconstruction_is_not_forced(self):
        data,d=self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            path=pathlib.Path(directory)/'paths.bin';path.write_bytes(data)
            fusion_verify.check_paths(path,d)
            struct.pack_into('<f',data,16,11);path.write_bytes(data)
            with self.assertRaises(ValueError):fusion_verify.check_paths(path,d)

    def test_v1_layout_still_reads(self):
        header=[1,1,1,9,1,1,1,0]+[0]*24
        data=bytearray(struct.pack('<32I',*header)+bytes(32+9*80))
        struct.pack_into('<4f',data,128,1,2,3,6)
        struct.pack_into('<4I',data,144,0,0,0,0x80000000)
        d=dict(schema_version=1,ray_stride=1,stored_rays=1,slots_per_ray=9,native_width=1,native_height=1,spp=1,diagnostic_flags=0)
        with tempfile.TemporaryDirectory() as directory:
            path=pathlib.Path(directory)/'v1.bin';path.write_bytes(data);fusion_verify.check_paths(path,d)


if __name__=='__main__':unittest.main()
