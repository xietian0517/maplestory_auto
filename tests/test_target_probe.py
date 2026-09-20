import ctypes as C
import unittest
from maplebot.target_probe import MemoryBasicInformation, search_private_memory


class FakeKernel:
    def __init__(self,size):
        self.size = size
    def VirtualQueryEx(self,handle,address,info,size):
        result = C.cast(info,C.POINTER(MemoryBasicInformation)).contents
        result.base = address
        result.region_size = self.size if address == 0 else 0x7fffffff0000-address
        result.state, result.protect = 0x1000,4
        result.type = 0x20000 if address == 0 else 0
        return size


class TargetProbeTests(unittest.TestCase):
    def test_utf16_hit_across_chunk_boundary(self):
        name = '小棒槌啊'
        data = b'a'*(1024*1024-3)+name.encode('utf-16-le')+b'b'*20
        class Reader:
            k = FakeKernel(len(data))
            def read(self,position,size):
                return data[position:position+size]
        report = search_private_memory(Reader(),1,[name])
        self.assertTrue(report['completed'])
        self.assertEqual(report['hits'],[dict(label=name,encoding='utf-16-le',address=hex(1024*1024-3))])

    def test_denied_reads_are_not_a_negative_search_result(self):
        class Reader:
            k = FakeKernel(40*1024*1024)
            def read(self,position,size):
                raise C.WinError(5)
        result = search_private_memory(Reader(),1,['小棒槌啊'])
        self.assertFalse(result['completed'])
        self.assertEqual(result['bytes_read'],0)
        self.assertEqual(result['read_errors'],32)


if __name__ == '__main__':
    unittest.main()
