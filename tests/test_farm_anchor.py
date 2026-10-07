import unittest
from autofarm.realtime.control import Controller
from autofarm.realtime.model import Actor,Box,MotionProfile,Observation,Platform
from autofarm.realtime.recovery import ActiveRecovery


class FarmAnchorTests(unittest.TestCase):
    def fixture(self):
        c=Controller(MotionProfile(180,100,150,True),True);c.last_epoch=0;c.no_enemy_since=0
        o=Observation(1,2,Actor(Box(285,52,315,100),.99),
            platforms=[Platform('a',0,600,100),Platform('b',0,600,200),Platform('c',0,600,300)],
            navigation_targets=[Actor(Box(310,150,350,200),.99)])
        return c,o

    def test_configured_anchor_outweighs_other_floor_but_not_current_combat(self):
        c,o=self.fixture();c.farm_anchor='c'
        self.assertEqual(c.decide(o,2).target,'c')
        c,o=self.fixture();c.farm_anchor='c';o.monsters=[Actor(Box(440,52,480,100),.99)]
        self.assertNotEqual(c.decide(o,2).target,'c')

    def test_wait_is_bounded_then_other_routes_resume(self):
        c,o=self.fixture();c.farm_anchor='a'
        d=c.decide(o,2)
        self.assertEqual(d.reason,'anchor_wait_respawn');self.assertFalse(d.keys)
        o.captured_at=6.1
        self.assertEqual(c.decide(o,6.1).target,'b')
        self.assertGreater(c.anchor_cooldown_until,6.1)

    def test_unknown_identity_and_forced_exploration_override_anchor(self):
        c,o=self.fixture();c.farm_anchor='a';o.player=None;o.reason='player_not_found'
        self.assertFalse(c.decide(o,2).keys)
        self.assertNotEqual(c.decide(o,2).reason,'anchor_wait_respawn')
        c,o=self.fixture();c.farm_anchor='a';c.request_exploration(o,2)
        self.assertNotEqual(c.decide(o,2).reason,'anchor_wait_respawn')

    def test_default_controller_keeps_existing_route(self):
        c,o=self.fixture()
        self.assertEqual(c.decide(o,2).target,'b')

    def test_runtime_recovery_does_not_walk_during_bounded_respawn_wait(self):
        c,o=self.fixture();c.farm_anchor='a';recovery=ActiveRecovery()
        for t in (2,3,4,5.9):
            o.captured_at=t;d=c.decide(o,t)
            final=recovery.apply(o,d,t,1366,controller=c)
            self.assertEqual(final.reason,'anchor_wait_respawn');self.assertFalse(final.keys)
        o.captured_at=6.1
        self.assertNotEqual(c.decide(o,6.1).reason,'anchor_wait_respawn')


if __name__=='__main__':unittest.main()
