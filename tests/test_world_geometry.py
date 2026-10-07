import unittest
from autofarm.realtime.model import Rope,Box
from scripts.build_world_geometry import merge_rope


class RopeRegistrationTests(unittest.TestCase):
    def test_only_clipped_endpoint_is_extended(self):
        ropes=[dict(x=100,top=200,bottom=693,top_clipped=False,bottom_clipped=True)]
        conflicts=[]
        merge_rope(ropes,Rope(100,0,430),(0,-400),Box(0,0,1366,694),conflicts)
        self.assertEqual(len(ropes),1)
        self.assertEqual((ropes[0]['top'],ropes[0]['bottom']),(200,830))
        self.assertFalse(ropes[0]['bottom_clipped'])

    def test_complete_rope_cannot_be_stretched_by_conflicting_annotation(self):
        ropes=[dict(x=100,top=200,bottom=500,top_clipped=False,bottom_clipped=False)]
        conflicts=[]
        merge_rope(ropes,Rope(101,200,620),(0,0),Box(0,0,1366,694),conflicts)
        self.assertEqual(ropes[0]['bottom'],500)
        self.assertEqual(conflicts[0]['reason'],'rope_bottom_disagreement')

    def test_shifted_duplicate_is_not_a_second_grabbable_rope(self):
        ropes=[dict(x=100,top=200,bottom=500,top_clipped=False,bottom_clipped=False)]
        conflicts=[]
        merge_rope(ropes,Rope(113,200,500),(0,0),Box(0,0,1366,694),conflicts)
        self.assertEqual(len(ropes),1);self.assertEqual(ropes[0]['x'],100)
        self.assertEqual(conflicts[0]['reason'],'rope_x_disagreement')


if __name__=='__main__':unittest.main()
