import unittest
from autofarm.realtime.control import Controller
from autofarm.realtime.model import Actor,Box,Decision,MotionProfile,Observation,Platform


class AttackOcclusionTests(unittest.TestCase):
    def observation(self,t,monsters=None,x=300,y=400):
        return Observation(1,t,Actor(Box(x-15,y-48,x+15,y),.99),
            monsters=list(monsters or []),platforms=[Platform('ground',0,1200,400)])

    def armed(self):
        c=Controller(MotionProfile(180,100,140,True));c.direct_attacks=True;c.last_epoch=0
        c.acknowledge(Decision(frozenset({'right'})),.8)
        target=Actor(Box(480,355,520,400),.99,track_id=7)
        o=self.observation(1,[target]);d=c.decide(o,1);c.acknowledge(d,1)
        return c,target

    def test_bounded_occlusion_preserves_acknowledged_hold_then_releases(self):
        c,_=self.armed();o=self.observation(1.1);d=c.decide(o,1.1)
        self.assertEqual(d.reason,'direct_attack_occlusion_hold')
        self.assertEqual(c.orient_attack(o,d,1.1).keys,{'shift'})
        self.assertNotIn('shift',c.decide(self.observation(1.19),1.19).keys)

    def test_cannot_start_without_a_submitted_attack(self):
        c=Controller();c.direct_attacks=True
        target=Actor(Box(480,355,520,400),.99,track_id=7)
        c.decide(self.observation(1,[target]),1)  # Proposal was never submitted.
        self.assertFalse(c.can_hold_occluded(self.observation(1.1),1.1))

    def test_current_crossing_evidence_and_player_knockback_cancel_grace(self):
        for monsters,x in (([Actor(Box(180,355,220,400),.99,track_id=7)],300),([],320)):
            c,_=self.armed();o=self.observation(1.1,monsters,x=x)
            self.assertFalse(c.can_hold_occluded(o,1.1))
            d=c.decide(o,1.1)
            self.assertNotEqual(d.reason,'direct_attack_occlusion_hold')
            self.assertNotIn('shift',c.orient_attack(o,d,1.1).keys)

    def test_lost_identity_cannot_use_target_memory(self):
        c,_=self.armed();o=self.observation(1.1);o.player=None;o.reason='player_not_found'
        self.assertFalse(c.decide(o,1.1).keys)

    def test_occluded_holds_do_not_renew_last_seen_timer(self):
        c,_=self.armed()
        for t in (1.05,1.1,1.15):
            d=c.decide(self.observation(t),t);c.acknowledge(d,t)
        self.assertNotIn('shift',c.decide(self.observation(1.2),1.2).keys)


if __name__=='__main__':unittest.main()
