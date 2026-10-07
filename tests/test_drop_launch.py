import unittest

from autofarm.realtime.control import Controller
from autofarm.realtime.model import Actor, Box, Observation, Platform, Rope


class DropLaunchTests(unittest.TestCase):
    def observation(self, x=120, vx=0):
        return Observation(1, 1, Actor(Box(x-15, 252, x+15, 300), .99, vx=vx),
                           platforms=[Platform('upper', 0, 600, 300),
                                      Platform('lower', 80, 600, 420)])

    def test_inside_overlap_drops_without_walking_to_midpoint(self):
        c=Controller(navigate=True)
        edge=('upper','lower','drop')
        d=c._navigate_step(self.observation(),edge,1)
        self.assertEqual(d.reason,'drop')
        self.assertEqual(d.keys,{'down'})
        self.assertEqual(c._navigate_step(self.observation(),edge,1.15).keys,{'down','alt'})
        self.assertFalse(c._navigate_step(self.observation(),edge,1.5).keys)

    def test_outside_overlap_still_approaches_supported_landing(self):
        d=Controller()._navigate_step(self.observation(x=30),('upper','lower','drop'),1)
        self.assertEqual(d.reason,'approach_launch')
        self.assertEqual(d.keys,{'right'})

    def test_horizontal_momentum_is_released_before_starting_drop(self):
        c=Controller();edge=('upper','lower','drop')
        d=c._navigate_step(self.observation(vx=190),edge,1)
        self.assertEqual(d.reason,'drop_brake');self.assertFalse(d.keys)
        self.assertIsNone(c.transition)
        # Finish braking even if inertia carries the player outside overlap;
        # do not alternate Left/Right before motion has settled.
        self.assertEqual(c._navigate_step(self.observation(x=65,vx=-120),edge,1.05).reason,'drop_brake')
        self.assertEqual(c._navigate_step(self.observation(vx=20),edge,1.1).reason,'drop_brake')
        self.assertEqual(c._navigate_step(self.observation(vx=-48),edge,1.36).keys,{'down'})
        # Once committed, velocity changes cannot restart the chord timer.
        self.assertEqual(c._navigate_step(self.observation(vx=100),edge,1.51,airborne=True).keys,{'down','alt'})

    def test_narrow_overlap_also_requires_braking(self):
        c=Controller();o=self.observation(x=130,vx=190)
        o.platforms[0]=Platform('upper',80,180,300)
        d=c._navigate_step(o,('upper','lower','drop'),1)
        self.assertEqual(d.reason,'drop_brake')
        self.assertFalse(d.keys)
        self.assertIsNone(c.transition)

    def test_changed_route_discards_old_braking_state(self):
        c=Controller();c.drop_brake_edge=('upper','other','drop')
        d=c._navigate_step(self.observation(),('upper','lower','drop'),1)
        self.assertEqual(d.reason,'drop');self.assertIsNone(c.drop_brake_edge)

    def test_drop_moves_off_rope_mouth_then_brakes_before_down(self):
        c=Controller();edge=('upper','lower','drop')
        o=self.observation(x=300);o.ropes=[Rope(300,300,400)]
        d=c._navigate_step(o,edge,1)
        self.assertEqual(d.reason,'approach_launch');self.assertNotIn('down',d.keys)
        o=self.observation(x=276,vx=-100);o.ropes=[Rope(300,300,400)]
        self.assertEqual(c._navigate_step(o,edge,1.1).reason,'drop_brake')
        o=self.observation(x=276);o.ropes=[Rope(300,300,400)]
        self.assertEqual(c._navigate_step(o,edge,1.2).reason,'drop_brake')
        self.assertEqual(c._navigate_step(o,edge,1.46).keys,{'down'})

    def narrow(self,x,vx=0):
        o=self.observation(x,vx)
        o.platforms=[Platform('upper',72,664,300),Platform('lower',347,453,472)]
        o.ropes=[Rope(157,300,485),Rope(604,300,420)]
        return o

    def test_reviewed_overshoot_can_drop_without_reverse(self):
        c=Controller();edge=('upper','lower','drop')
        self.assertEqual(c._navigate_step(self.narrow(440,-160),edge,1).reason,'approach_launch')
        self.assertEqual(c._navigate_step(self.narrow(430,-160),edge,1.05).reason,'drop_brake')
        self.assertEqual(c._navigate_step(self.narrow(405.5,-198),edge,1.1).reason,'drop_brake')
        self.assertEqual(c._navigate_step(self.narrow(373,-127),edge,1.3).reason,'drop_brake')
        d=c._navigate_step(self.narrow(374,48),edge,1.56)
        self.assertEqual(d.keys,{'down'});self.assertEqual(d.reason,'drop')

    def test_settled_outside_safe_landing_must_reapproach(self):
        c=Controller();edge=('upper','lower','drop')
        c.drop_brake_edge=edge
        c._navigate_step(self.narrow(360),edge,1)
        d=c._navigate_step(self.narrow(360),edge,1.26)
        self.assertEqual(d.reason,'approach_launch');self.assertEqual(d.keys,{'right'})

    def test_settled_rope_mouth_still_requires_clearance(self):
        c=Controller();edge=('upper','lower','drop');c.drop_brake_edge=edge
        o=self.narrow(400);o.ropes=[Rope(400,300,480)]
        c._navigate_step(o,edge,1)
        d=c._navigate_step(o,edge,1.26)
        self.assertEqual(d.reason,'approach_launch');self.assertNotIn('down',d.keys)

    def test_generic_recovery_does_not_interrupt_owned_drop_wait(self):
        from autofarm.realtime.recovery import ActiveRecovery
        from autofarm.realtime.model import Decision
        c=Controller();c.drop_brake_edge=('upper','lower','drop');r=ActiveRecovery()
        for t in (1,1.5,2,2.5):
            o=self.narrow(400);o.captured_at=t
            d=r.apply(o,Decision(reason='drop_brake'),t,1366,controller=c)
            self.assertEqual(d.reason,'drop_brake');self.assertFalse(d.keys)

    def test_no_clear_supported_launch_cannot_issue_down(self):
        c=Controller();o=self.observation(x=130)
        o.platforms[0]=Platform('upper',110,150,300);o.ropes=[Rope(130,300,400)]
        d=c._navigate_step(o,('upper','lower','drop'),1)
        self.assertEqual(d.reason,'drop_rope_blocked');self.assertFalse(d.keys)
        self.assertIsNone(c.transition)


if __name__=='__main__':unittest.main()
