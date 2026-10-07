import unittest
from autofarm.realtime.control import Controller
from autofarm.realtime.model import Actor,Box,Decision,Observation,Platform


class FacingRefreshTests(unittest.TestCase):
    def test_cached_direction_reassert_keeps_the_standing_attack(self):
        c=Controller();c.refresh_attack_facing=True;c.applied_facing='right';c.applied_face_at=0
        d=Decision(frozenset({'shift'}),'attack','1')
        def obs(t):return Observation(1,t,Actor(Box(620,584,650,632),.99),
            [Actor(Box(685,588,734,634),.99,track_id=1)],[Platform('ground',0,1366,638)])
        # The direction model already matches; the re-assert must not cost the
        # attack, and the window must still end on its own.
        for t in (1.,1.05,1.1,1.17):
            action=c.orient_attack(obs(t),d,t)
            self.assertEqual(action.keys,{'right','shift'});c.acknowledge(action,t)
        self.assertEqual(c.orient_attack(obs(1.19),d,1.19).keys,{'shift'})
        self.assertEqual(c.orient_attack(obs(2.41),d,2.41).keys,{'right','shift'})

    def test_jump_attack_and_real_turn_still_release_the_attack_key(self):
        c=Controller();c.refresh_attack_facing=True;c.applied_facing='right'
        o=Observation(1,1,Actor(Box(620,584,650,632),.99),
            [Actor(Box(685,588,734,634),.99,track_id=1)],[Platform('ground',0,1366,638)])
        jump=Decision(frozenset({'shift'}),'jump_attack_fire_first','1')
        self.assertEqual(c.orient_attack(o,jump,1).keys,{'right'})
        c.applied_facing='left'
        self.assertEqual(c.orient_attack(o,Decision(frozenset({'shift'}),'attack','1'),1).keys,{'right'})

    def test_refresh_does_not_move_off_edge_or_turn_without_target(self):
        c=Controller();c.refresh_attack_facing=True;c.applied_facing='right'
        o=Observation(1,1,Actor(Box(620,584,650,632),.99),
            [Actor(Box(685,588,734,634),.99,track_id=1)],[Platform('ledge',600,650,638)])
        d=Decision(frozenset({'shift'}),'attack','1')
        self.assertNotIn('right',c.orient_attack(o,d,1).keys)
        o.monsters=[];self.assertFalse(c.orient_attack(o,d,1).keys)
