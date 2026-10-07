"""Regressions for control hazards found while reviewing human transfers."""
import unittest
from types import SimpleNamespace

from autofarm.realtime.control import Controller,graph
from autofarm.realtime.model import Actor, Box, Decision, Observation, Platform,Rope,MotionProfile
from autofarm.realtime.climbing import RopeClimber


class TransferPriorityTests(unittest.TestCase):
    def test_held_attack_rechecks_missing_crossed_and_close_targets(self):
        for enemy_x in (None,100,225):
            with self.subTest(enemy_x=enemy_x):
                c=Controller(navigate=True);c.last_epoch=0;c.direct_attacks=True
                c.applied_facing='right';c.applied_face_at=0
                c.direct_hold_until=3;c.direct_rest_until=3.15;c.direct_target='7'
                monsters=[] if enemy_x is None else [Actor(Box(enemy_x-15,160,enemy_x+15,200),1,0,0,7)]
                o=Observation(1,1,Actor(Box(185,150,215,200),1),monsters=monsters,
                              platforms=[Platform('ground',0,500,200)])
                d=c.orient_attack(o,c.decide(o,1),1)
                self.assertNotIn('shift',d.keys)
                self.assertEqual(c.direct_hold_until,0)
                if enemy_x==100:self.assertEqual(d.keys,{'left'})
                if enemy_x==225:self.assertEqual(d.reason,'ranged_make_room')

    def test_nearby_enemy_cannot_cancel_attached_rope_at_low_speed(self):
        for direct in (False, True):
            for near_floor in (False, True):
                with self.subTest(direct=direct, near_floor=near_floor):
                    c=Controller(navigate=True);c.last_epoch=0;c.direct_attacks=direct
                    c.direct_hold_until=3;c.direct_rest_until=3.15;c.direct_target='7'
                    climb=SimpleNamespace(phase='ascend', decide=lambda o,t:Decision(frozenset({'up'}),'rope_ascend','upper'))
                    c.rope_climber=climb
                    floor=Platform('lower',0,400,200 if near_floor else 350)
                    o=Observation(1,1,player=Actor(Box(180,140,220,200),1,0,0),
                                  monsters=[Actor(Box(280,160,310,200),1,0,0,7)],platforms=[floor])
                    d=c.decide(o,1)
                    self.assertEqual(d.keys,frozenset({'up'}))
                    self.assertEqual(d.reason,'rope_ascend')
                    self.assertIs(c.rope_climber,climb)
                    self.assertEqual(c.direct_hold_until,0)

    def test_normal_ground_combat_remains_available(self):
        c=Controller(navigate=True);c.direct_attacks=True;c.last_epoch=0
        o=Observation(1,1,player=Actor(Box(180,140,220,200),1),
                      monsters=[Actor(Box(280,160,310,200),1,0,0,7)],
                      platforms=[Platform('ground',0,500,200)])
        self.assertEqual(c.decide(o,1).reason,'direct_attack_start')

    def test_ranged_spacing_uses_floor_room_but_never_retreats_over_edge(self):
        for x,expected in [(200,'ranged_make_room'),(25,'direct_attack_start')]:
            c=Controller(navigate=True);c.direct_attacks=True;c.last_epoch=0
            o=Observation(1,1,player=Actor(Box(x-15,150,x+15,200),1),
                monsters=[Actor(Box(x+20,160,x+40,200),1,0,0,7)],platforms=[Platform('ground',0,500,200)])
            self.assertEqual(c.decide(o,1).reason,expected)

    def test_same_floor_distant_target_is_approached_before_rope_exploration(self):
        c=Controller(navigate=True);c.last_epoch=0;c.no_enemy_since=0
        o=Observation(1,2,player=Actor(Box(180,150,210,200),1),
            navigation_targets=[Actor(Box(800,150,830,200),1)],platforms=[Platform('ground',0,1200,200)])
        d=c.decide(o,2)
        self.assertEqual(d.reason,'approach');self.assertEqual(d.keys,frozenset({'right'}))

    def test_rope_crossing_source_platform_can_be_grabbed_upward(self):
        ps=[Platform('lower',0,400,500),Platform('upper',0,400,200)]
        rope=Rope(200,200,650)
        self.assertIn(('upper','rope'),graph(ps,[rope],MotionProfile(200,110,100,True))['lower'])
        o=Observation(1,1,Actor(Box(185,450,215,500),.99),platforms=ps,ropes=[rope])
        d=RopeClimber(target_id='upper',jump_height=110).decide(o,1)
        self.assertEqual(d.reason,'rope_catch');self.assertIn('up',d.keys)
        below=Rope(200,510,650)
        self.assertNotIn(('upper','rope'),graph(ps,[below],MotionProfile(200,110,100,True))['lower'])


if __name__=='__main__':
    unittest.main()
