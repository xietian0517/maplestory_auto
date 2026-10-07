import unittest
from autofarm.realtime.control import Controller
from autofarm.realtime.model import Actor,Box,Decision,MotionProfile,Observation,Platform


class StandingCommitTests(unittest.TestCase):
    def observation(self,t,left=80,blocker=False):
        monsters=[Actor(Box(480,355,520,400),.99,track_id=1),
                  Actor(Box(left-20,355,left+20,400),.99,track_id=2)]
        if blocker:monsters.append(Actor(Box(330,355,370,400),.99,track_id=3))
        return Observation(1,t,Actor(Box(285,352,315,400),.99),monsters,
                           [Platform('ground',0,900,400)])

    def armed(self):
        c=Controller(MotionProfile(180,110,140,True));c.last_epoch=0
        c.acknowledge(Decision(frozenset({'right'})),.8)
        d=c.decide(self.observation(1),1);c.acknowledge(d,1)
        self.assertEqual(d.target,'1');return c

    def test_visible_original_direction_survives_nearer_opposite_enemy_for_half_second(self):
        c=self.armed()
        for t in (1.1,1.2,1.3,1.4,1.49):
            d=c.decide(self.observation(t,left=120),t)
            self.assertEqual(d.target,'1');self.assertEqual(d.keys,{'shift'});c.acknowledge(d,t)
        d=c.decide(self.observation(1.51,left=120),1.51)
        self.assertEqual(d.keys,{'left'})

    def test_close_enemy_immediately_breaks_direction_commit(self):
        c=self.armed();d=c.decide(self.observation(1.1,left=120,blocker=True),1.1)
        self.assertEqual(d.keys,{'left'})

    def test_proposed_attack_cannot_start_commit(self):
        c=Controller();c.last_epoch=0;c.applied_facing='right';c.facing='right'
        c.decide(self.observation(1),1)
        self.assertEqual(c.decide(self.observation(1.1,left=120),1.1).keys,{'left'})

    def test_missing_target_or_identity_never_uses_commit_for_blind_attack(self):
        c=self.armed();o=self.observation(1.1);o.monsters=[]
        self.assertNotIn('shift',c.decide(o,1.1).keys)
        o=self.observation(1.2);o.player=None;o.reason='player_not_found'
        self.assertFalse(c.decide(o,1.2).keys)

    def test_input_gap_and_reset_expire_commit(self):
        for reset in (False,True):
            c=self.armed()
            if reset:c.reset();c.last_epoch=0;c.applied_facing='right';c.facing='right'
            self.assertEqual(c.decide(self.observation(1.3,left=120),1.3).keys,{'left'})

    def test_short_detection_gap_pauses_navigation_without_blind_fire(self):
        c=self.armed();o=self.observation(1.1);o.monsters=[]
        d=c.decide(o,1.1)
        self.assertEqual(d.reason,'attack_reacquire_wait');self.assertFalse(d.keys)
        c.acknowledge(d,1.1)
        o=self.observation(1.19);o.monsters=[]
        self.assertNotEqual(c.decide(o,1.19).reason,'attack_reacquire_wait')

    def test_navigation_pause_requires_accepted_attack_and_stable_identity(self):
        c=Controller();c.last_epoch=0;c.applied_facing='right';c.facing='right'
        c.decide(self.observation(1),1)
        o=self.observation(1.1);o.monsters=[]
        self.assertFalse(c.wait_standing_target(o,1.1))
        c=self.armed();o.player=Actor(Box(305,352,335,400),.99)
        self.assertFalse(c.wait_standing_target(o,1.1))
        o=self.observation(1.1);o.reason='identity_confirming'
        self.assertFalse(c.wait_standing_target(o,1.1))

    def test_reappearing_target_resumes_normal_attack_before_grace_expires(self):
        c=self.armed();o=self.observation(1.05);o.monsters=[]
        self.assertEqual(c.decide(o,1.05).reason,'attack_reacquire_wait')
        d=c.decide(self.observation(1.1),1.1)
        self.assertEqual(d.keys,{'shift'});self.assertEqual(d.target,'1')


if __name__=='__main__':unittest.main()
