import ctypes as C
import os
import subprocess
import sys
import unittest
from maplebot.memory import ProcessReader


@unittest.skipUnless(sys.platform == 'win32', 'Windows API only')
class MemoryTests(unittest.TestCase):
    def test_read_other_process(self):
        # Windows venv python.exe can be a launcher with a different process ID.
        script = "import ctypes as c,os; x=c.c_int32(314159); print(os.getpid(),c.addressof(x),flush=True); input()"
        with subprocess.Popen([sys.executable, '-c', script], stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE, text=True) as child:
            try:
                pid, address = map(int, child.stdout.readline().split())
                with ProcessReader(pid) as reader:
                    self.assertEqual(reader.scalar(address, 'i32'), 314159)
                    self.assertTrue(reader.modules())
            finally:
                child.communicate('\n', timeout=5)

    def test_pointer_chain_and_closed_reader(self):
        value = C.c_float(12.5)
        pointer = C.c_void_p(C.addressof(value))
        with ProcessReader(os.getpid()) as reader:
            address = reader.resolve(C.addressof(pointer), [0], C.sizeof(C.c_void_p))
            self.assertEqual(reader.scalar(address), 12.5)
            with self.assertRaises(ValueError):
                reader.read(0, 4)
            with self.assertRaises(OSError):
                reader.read(1, 4)
        with self.assertRaises(RuntimeError):
            reader.read(address, 4)

    def test_null_pointer_rejected(self):
        pointer = C.c_void_p()
        with ProcessReader(os.getpid()) as reader:
            with self.assertRaises(ValueError):
                reader.resolve(C.addressof(pointer), [16], C.sizeof(C.c_void_p))


if __name__ == '__main__':
    unittest.main()
