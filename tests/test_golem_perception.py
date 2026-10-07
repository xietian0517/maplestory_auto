"""Regressions from the stone-golem temple's zero-attack recordings."""
from pathlib import Path
import unittest

import cv2

from autofarm.realtime.minimap import locate_minimap, yellow_points
from autofarm.realtime.model import Actor, Box, Scene
from autofarm.realtime.perception import GroundedVision


class GolemPerceptionTests(unittest.TestCase):
    def setUp(self):
        cv2.setNumThreads(2)
        self.image = cv2.imread(str(Path(__file__).parent/'fixtures/realtime/stone_golem_seed.png'))

    def test_same_floor_tall_golem_is_detected_in_combat_region(self):
        scene = Scene.parse(dict(request_id='golem', map_name='temple', width=1366, height=768,
            play_area=[0,0,1366,613], player_name_box=[580,468,627,480], player_foot_offset=-13,
            monster_boxes=[[1068,126,1246,280]], exclude_boxes=[],
            platforms=[dict(id='p1',left=432,right=1207,y=281)], ropes=[],
            preferred_platforms=['p1'], confidence=.9, reasoning='reviewed frame'), 'golem',1366,768)
        vision = GroundedVision(scene,self.image)
        try:
            player = Actor(Box(885,233,915,281),.99)
            hits = vision.detect_monsters(self.image,vision.combat_roi(player),player)
            self.assertTrue(any(abs(h.box.cx-1157)<3 and abs(h.box.y2-280)<3 for h in hits))
        finally:
            vision.close()

    def test_temple_minimap_with_bright_stone_terrain_is_located(self):
        roi = locate_minimap(self.image)
        self.assertIsNotNone(roi)
        x,y,w,h = roi
        points = yellow_points(self.image[y:y+h,x:x+w])
        self.assertEqual(len(points),1)
        self.assertAlmostEqual(points[0][0]+x,50.5,delta=1)
        self.assertAlmostEqual(points[0][1]+y,119.5,delta=1)

    def test_walking_golem_is_detected_without_matching_door_statues(self):
        scene = Scene.parse(dict(request_id='golem', map_name='temple', width=1366, height=768,
            play_area=[0,0,1366,613], player_name_box=[580,468,627,480], player_foot_offset=-13,
            monster_boxes=[[1068,126,1246,280]], exclude_boxes=[],
            platforms=[dict(id='p1',left=0,right=1366,y=461)], ropes=[],
            preferred_platforms=['p1'], confidence=.9, reasoning='reviewed frame'), 'golem',1366,768)
        vision = GroundedVision(scene,self.image)
        try:
            walking = cv2.imread(str(Path(__file__).parent/'fixtures/realtime/stone_golem_walking.png'))
            hits = vision.detect_monsters(walking,scene.play_area)
            self.assertTrue(any(abs(h.box.cx-640)<12 and abs(h.box.y2-461)<8 for h in hits))
            self.assertFalse(any(390<h.box.cx<480 and 330<h.box.y2<470 for h in hits))
            self.assertFalse(any(520<h.box.cx<600 and h.box.y2<100 for h in hits))
        finally:
            vision.close()


if __name__ == '__main__':
    unittest.main()
