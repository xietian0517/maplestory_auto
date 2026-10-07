import json
from pathlib import Path
import unittest
from autofarm.realtime.control import Controller,standing_platform
from autofarm.realtime.model import Actor,Box,Observation,Platform


class RightLeafGeometryTests(unittest.TestCase):
    def observation(self,x,y,dy,vy=0):
        path=Path(__file__).resolve().parents[1]/'dist/captures/monkey_forest_v043/world_geometry.json'
        data=json.loads(path.read_text(encoding='utf-8'))
        return Observation(1,1,Actor(Box(x-15,y-48,x+15,y),.99,vy=vy),
            platforms=[Platform(p['id'],p['left'],p['right'],p['y']+dy) for p in data['platforms']])

    def test_recorded_stationary_positions_are_supported(self):
        for x,y,dy in [(982,420,-166),(1016,458,-134)]:
            o=self.observation(x,y,dy)
            self.assertEqual(standing_platform(o).id,'reviewed_right_upper_leaf')
            self.assertNotEqual(Controller().decide(o,1).reason,'airborne_or_floor_unknown')

    def test_old_boundary_reproduces_the_reported_failure(self):
        o=self.observation(982,420,-166)
        o.platforms=[Platform(p.id,1036,p.right,p.y) if p.id=='reviewed_right_upper_leaf' else p for p in o.platforms]
        self.assertIsNone(standing_platform(o))

    def test_falling_and_real_gap_are_not_grounded(self):
        self.assertIsNone(standing_platform(self.observation(982,420,-166,vy=200)))
        self.assertIsNone(standing_platform(self.observation(945,420,-166)))
