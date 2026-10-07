import unittest
from autofarm.realtime.control import Controller
from autofarm.realtime.model import Actor,Box,Decision,Observation,Platform


def observed(t,monsters=None):
    return Observation(1,t,Actor(Box(285,352,315,400),.99),
        monsters if monsters is not None else [Actor(Box(480,410,520,455),.99,track_id=1)],
        [Platform('leaf',240,360,400),Platform('ground',0,900,465)])


class MonkeyBurstTests(unittest.TestCase):
    def armed(self):
        c=Controller();c.last_epoch=0;c.standing_burst_seconds=1.4;c.standing_continue_occluded=True
        c.acknowledge(Decision(frozenset({'right'})),.8)
        d=c.decide(observed(1),1);c.acknowledge(d,1)
        self.assertEqual(d.keys,{'shift'})
        return c

    def test_continues_short_gap_without_renewing_visibility_or_inventing_kill(self):
        c=self.armed()
        for t in (1.04,1.1,1.17):
            d=c.decide(observed(t,[]),t)
            self.assertEqual(d.keys,{'shift'});self.assertEqual(d.reason,'attack_burst_continue')
            self.assertEqual(c.orient_attack(observed(t,[]),d,t),d)
            c.acknowledge(d,t)
        self.assertNotIn('shift',c.decide(observed(1.19,[]),1.19).keys)

    def test_commit_keeps_visible_firing_side_for_longer_pair_window(self):
        c=self.armed()
        for i in range(1,14):
            t=1+i*.1;o=observed(t)
            o.monsters.append(Actor(Box(140,410,180,455),.99,track_id=2))
            d=c.decide(o,t);self.assertEqual(d.target,'1');self.assertEqual(d.keys,{'shift'})
            c.acknowledge(d,t)

    def test_close_contact_identity_loss_and_displacement_interrupt(self):
        for kind in ('contact','identity','motion','reset'):
            c=self.armed();o=observed(1.1,[])
            if kind=='contact':o.monsters=[Actor(Box(315,352,345,400),.99,track_id=3)]
            elif kind=='identity':o.reason='player_not_found';o.player=None
            elif kind=='motion':o.player=Actor(Box(315,352,345,400),.99)
            else:c.reset()
            self.assertFalse(c.can_continue_standing(o,1.1))

    def test_unaccepted_proposal_does_not_hold(self):
        c=Controller();c.standing_continue_occluded=True;c.last_epoch=0
        c.applied_facing=c.facing='right';c.decide(observed(1),1)
        self.assertFalse(c.can_continue_standing(observed(1.1,[]),1.1))

    def test_firing_anchor_needs_targets_empty_ledge_and_route(self):
        c=Controller();c.farm_anchor='leaf';o=observed(1)
        edges={'ground':[('leaf','jump')]};floor=o.platforms[1]
        self.assertTrue(c.firing_anchor_available(o,edges,floor))
        self.assertFalse(c.firing_anchor_available(o,{},floor))
        o.monsters.append(Actor(Box(290,355,330,400),.99,track_id=2))
        self.assertFalse(c.firing_anchor_available(o,edges,floor))
        o.monsters=[];self.assertFalse(c.firing_anchor_available(o,edges,floor))


if __name__=='__main__':unittest.main()
