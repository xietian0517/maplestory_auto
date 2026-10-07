import unittest
from autofarm.realtime.control import Controller
from autofarm.realtime.model import Actor,Box,MotionProfile,Observation,Platform


class NavigationWaitTests(unittest.TestCase):
    def observation(self,targets=(),platforms=None):
        return Observation(1,1,Actor(Box(285,352,315,400),.99),
            platforms=platforms or [Platform('ground',0,1200,400)],navigation_targets=list(targets))

    def controller(self):
        return Controller(MotionProfile(180,100,140,True),True)

    def test_fresh_distant_enemy_on_continuous_floor_does_not_wait(self):
        enemy=Actor(Box(780,355,820,400),.99)
        d=self.controller().decide(self.observation([enemy]),1)
        self.assertEqual(d.reason,'approach');self.assertEqual(d.keys,{'right'})

    def test_other_floor_enemy_requires_a_real_route(self):
        enemy=Actor(Box(310,455,350,500),.99)
        floors=[Platform('ground',0,600,400),Platform('lower',0,600,500)]
        self.assertNotEqual(self.controller().decide(self.observation([enemy],floors),1).reason,'search')
        floors[1]=Platform('unreachable',1000,1200,500)
        far=Actor(Box(1080,455,1120,500),.99)
        self.assertEqual(self.controller().decide(self.observation([far],floors),1).reason,'search')

    def test_empty_or_low_confidence_view_retains_search_delay(self):
        self.assertEqual(self.controller().decide(self.observation(),1).reason,'search')
        weak=Actor(Box(310,455,350,500),.6)
        floors=[Platform('ground',0,600,400),Platform('lower',0,600,500)]
        self.assertEqual(self.controller().decide(self.observation([weak],floors),1).reason,'search')

    def test_airborne_transition_does_not_start_next_approach_clock(self):
        c=self.controller();edge=('ground','lower','drop')
        o=self.observation(platforms=[Platform('ground',0,600,400),Platform('lower',0,600,500)])
        c._navigate_step(o,edge,1)
        self.assertIsNone(c.pending_since)
        c._navigate_step(o,edge,1.2,airborne=True)
        self.assertIsNone(c.pending_since)

    def test_new_route_does_not_inherit_elapsed_approach_budget(self):
        c=self.controller();c.last_epoch=0;c.pending_since=1
        o=self.observation([Actor(Box(510,455,550,500),.99)],
            [Platform('ground',0,600,400),Platform('lower',450,600,500)])
        o.captured_at=10
        self.assertEqual(c.decide(o,10).reason,'approach_launch')
        o.captured_at=10.1
        self.assertEqual(c.decide(o,10.1).reason,'approach_launch')
        self.assertNotIn(('ground','lower'),c.failures)


if __name__=='__main__':unittest.main()
