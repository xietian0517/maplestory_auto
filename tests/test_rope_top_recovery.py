import unittest
from autofarm.realtime.control import Controller
from autofarm.realtime.model import Actor,Box,Observation,Platform,Rope


class RopeTopRecoveryTests(unittest.TestCase):
    def observation(self,t=1,vy=0,x=170,y=666):
        return Observation(1,t,Actor(Box(x-15,y-48,x+15,y),.99,vy=vy),
            platforms=[Platform('ledge',0,285,648)],ropes=[Rope(169,648,773)],
            monsters=[Actor(Box(270,603,310,648),.99,track_id=4)])

    def test_stalled_18px_below_rope_top_overrides_visible_attack(self):
        for direct in (False,True):
            c=Controller(navigate=True);c.direct_attacks=direct
            d=c.decide(self.observation(),1)
            self.assertEqual(d.reason,'rope_observe_attachment');self.assertFalse(d.keys)
            d=c.decide(self.observation(1.16),1.16)
            self.assertEqual(d.reason,'rope_probe_attachment');self.assertEqual(d.keys,{'up'})
            self.assertFalse(c.rope_climber.grab_confirmed)

    def test_falling_or_off_rope_does_not_trigger_probe(self):
        for o in (self.observation(vy=90),self.observation(x=200)):
            c=Controller();c.decide(o,1);o.captured_at=1.2
            self.assertNotEqual(c.decide(o,1.2).reason,'rope_probe_attachment')
            self.assertIsNone(c.rope_climber)

    def test_failed_probe_blocks_repeated_up_at_same_position(self):
        c=Controller();c.decide(self.observation(),1);c.decide(self.observation(1.16),1.16)
        d=c.decide(self.observation(1.83),1.83)
        self.assertEqual(d.reason,'rope_resume_unconfirmed');self.assertFalse(d.keys)
        d=c.decide(self.observation(2),2)
        self.assertNotIn('up',d.keys);self.assertIsNone(c.rope_climber)


if __name__=='__main__':unittest.main()
