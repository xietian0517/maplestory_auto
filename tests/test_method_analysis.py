import unittest
try:
    from maplebot.method_analysis import method_span
except ModuleNotFoundError:
    method_span = None


@unittest.skipIf(method_span is None,'optional capstone dependency not installed')
class MethodSpanTests(unittest.TestCase):
    def test_shared_exception_range_does_not_merge_adjacent_getters(self):
        end,_ = method_span(0x1100,[(0x1000,0x2000)],[0x1000],[0x1100,0x1110])
        self.assertEqual(end,0x1110)

    def test_missing_exception_entry_is_bounded(self):
        self.assertEqual(method_span(0x1000,[],[],[0x1000,0x9000])[0],0x1100)

    def test_earlier_exception_end_wins(self):
        self.assertEqual(method_span(0x1000,[(0x1000,0x1020)],[0x1000],[0x1000,0x1100])[0],0x1020)


if __name__ == '__main__':
    unittest.main()
