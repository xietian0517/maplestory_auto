import unittest

from autofarm.realtime.control import Controller
from autofarm.realtime.model import Actor,Box,MotionProfile,Observation,Platform,Rope


class VisibilityGuardTests(unittest.TestCase):
    def setUp(self):
        self.c=Controller(MotionProfile(180,100,140,True),True)
        self.c.hud_boxes=[Box(1200,614,1366,694)]
        self.c.direct_attacks=True

    def observation(self,x=1160,y=637,vx=0,vy=0,platforms=None):
        return Observation(1,1,Actor(Box(x-15,y-48,x+15,y),.98,vx,vy),
            monsters=[Actor(Box(1270,592,1320,637),.99,track_id=4)],
            platforms=platforms if platforms is not None else [Platform('ground',0,1366,637)])

    def test_retreat_before_hotbar_hides_identity_even_with_visible_enemy(self):
        d=self.c.decide(self.observation(),1)
        self.assertEqual(d.reason,'avoid_hud_occlusion');self.assertEqual(d.keys,{'left'})
        # Continue out to the clear margin rather than oscillating at entry.
        d=self.c.decide(self.observation(x=1130),1)
        self.assertEqual(d.keys,{'left'})
        d=self.c.decide(self.observation(x=1115),1)
        self.assertNotEqual(d.reason,'avoid_hud_occlusion')

    def test_no_blind_movement_when_identity_or_floor_is_unknown(self):
        o=self.observation();o.player=None;o.reason='player_not_found'
        self.assertFalse(self.c.decide(o,1).keys)
        self.assertIsNone(self.c.avoid_hud(self.observation(platforms=[]),None))

    def test_does_not_walk_off_a_short_ledge_or_interrupt_rope(self):
        o=self.observation(platforms=[Platform('short',1140,1366,637)])
        self.assertIsNone(self.c.avoid_hud(o,o.platforms[0]))
        self.c.rope_climber=object()
        o=self.observation()
        self.assertIsNone(self.c.avoid_hud(o,o.platforms[0]))

    def test_screen_fixed_hud_does_not_affect_higher_floor_or_airborne_sprite(self):
        for o in (self.observation(y=450,platforms=[Platform('upper',0,1366,450)]),
                  self.observation(vy=-100)):
            self.assertIsNone(self.c.avoid_hud(o,o.platforms[0]))

    def test_map_change_discards_old_foothold_destination(self):
        self.c.decide(self.observation(),1)
        o=self.observation(x=1100,platforms=[Platform('new',1000,1190,637)]);o.map_epoch=1
        self.assertNotEqual(self.c.decide(o,1).reason,'avoid_hud_occlusion')


if __name__=='__main__':unittest.main()
