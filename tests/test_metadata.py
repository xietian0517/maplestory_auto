import struct
import unittest
from maplebot.metadata import Metadata, SECTIONS, index_width
from maplebot.pe_analysis import NativeTypes


def fixture():
    strings = bytearray()
    indices = {}
    for value in ('', 'Example', 'position', 'Update', 'Assembly-CSharp.dll'):
        indices[value] = len(strings)
        strings.extend(value.encode()+b'\0')
    pack = lambda fmt,*values: struct.pack('<'+fmt,*values)
    tables = {
        'string':(bytes(strings),len(indices)),
        'fields':(pack('iBI',indices['position'],0,0x04000001),1),
        'methods':(pack('iBBIBBIHHHH',indices['Update'],0,0,0,255,255,0x06000001,0,0,0,0),1),
        'typeDefinitions':(pack('iiBBBBIiiiiiiiiHHHHHHHHII',indices['Example'],0,0,255,255,255,1,
                                0,0,-1,-1,-1,-1,-1,-1,1,0,1,0,0,0,0,0,0,0x02000001),1),
        'images':(pack('iiBIBIiIiI',indices['Assembly-CSharp.dll'],0,0,1,255,0,-1,1,0,0),1),
        'stringLiteral':(pack('ii',0,5),2),
        'stringLiteralData':(b'hello',1),
    }
    data = bytearray(8+12*len(SECTIONS))
    struct.pack_into('<II',data,0,0xFAB11BAF,39)
    for i,name in enumerate(SECTIONS):
        content,count = tables.get(name,(b'',0))
        struct.pack_into('<iii',data,8+12*i,len(data),len(content),count)
        data.extend(content)
    return bytes(data)


class MetadataTests(unittest.TestCase):
    def test_v39_variable_width_member_ownership(self):
        m = Metadata(fixture())
        types = m.decode()
        self.assertEqual(types[0]['name'],'Example')
        self.assertEqual(types[0]['fields'][0]['name'],'position')
        self.assertEqual(types[0]['methods'][0]['name'],'Update')
        self.assertEqual(types[0]['assembly'],'Assembly-CSharp.dll')
        self.assertEqual(m.literals()[0]['value'],'hello')

    def test_index_width_null_sentinel_boundary(self):
        self.assertEqual([index_width(n) for n in (255,256,65535,65536)], [1,2,2,4])

    def test_wrong_version_and_out_of_bounds_section(self):
        data = bytearray(fixture())
        struct.pack_into('<I',data,4,29)
        with self.assertRaises(ValueError):
            Metadata(data)
        struct.pack_into('<I',data,4,39)
        struct.pack_into('<i',data,8,len(data)+1)
        with self.assertRaises(ValueError):
            Metadata(data)

    def test_wrong_record_size_rejected(self):
        m = Metadata(fixture())
        m.sections['fields']['size'] += 1
        with self.assertRaises(ValueError):
            m.decode()

    def test_native_primitive_flags(self):
        # Validate field attributes and primitive type tag independently.
        class Binary:
            data = struct.pack('<QI',0,(12<<16)|16)
            def raw(self,va,size):
                return 0
        native = NativeTypes.__new__(NativeTypes)
        native.pe, native.cache = Binary(), {}
        result = native.at(4096)
        self.assertEqual(result['name'],'System.Single')
        self.assertEqual(result['attributes'],16)


if __name__ == '__main__':
    unittest.main()
