import unittest
from autofarm.realtime.control import Controller
from autofarm.realtime.model import Actor,Box,MotionProfile,Observation,Platform


class JumpLaunchTests(unittest.TestCase):
    def case(self,x=669,vx=131):
        c=Controller(MotionProfile(164,113,112,True),True)
        o=Observation(1,1,Actor(Box(x-15,408,x+15,458),.99,vx=vx),
            platforms=[Platform('source',656,787,465),Platform('dest',571,698,406)])
        return c,o,('source','dest','jump')

    def test_round056_wrong_way_momentum_is_braked_before_overlap_jump(self):
        c,o,edge=self.case()
        d=c._navigate_step(o,edge,1)
        self.assertEqual(d.reason,'jump_brake');self.assertFalse(d.keys)
        self.assertIsNone(c.transition)
        o.player=Actor(Box(665,408,695,458),.99,vx=65)
        self.assertEqual(c._navigate_step(o,edge,1.1).reason,'jump_brake')
        o.player=Actor(o.player.box,.99,vx=0)
        d=c._navigate_step(o,edge,1.3)
        self.assertEqual(d.reason,'jump_brake')
        d=c._navigate_step(o,edge,1.51)
        self.assertEqual(d.reason,'jump');self.assertIn('alt',d.keys)

    def test_airborne_jump_is_not_replaced_by_ground_braking(self):
        c,o,edge=self.case();c.transition=edge;c.transition_started=.8
        d=c._navigate_step(o,edge,1,airborne=True)
        self.assertEqual(d.reason,'jump')

    def test_drift_outside_overlap_requires_reapproach(self):
        c,o,edge=self.case();c._navigate_step(o,edge,1)
        o.player=Actor(Box(692,408,722,458),.99,vx=0)
        c._navigate_step(o,edge,1.3)
        d=c._navigate_step(o,edge,1.56)
        self.assertEqual(d.reason,'approach_launch');self.assertNotIn('alt',d.keys)

    def test_stationary_pose_jitter_does_not_restart_braking_indefinitely(self):
        c,o,edge=self.case();c._navigate_step(o,edge,1)
        for t,x,vx in [(1.1,680,60),(1.2,681,-60),(1.3,679,60)]:
            o.player=Actor(Box(x-15,408,x+15,458),.99,vx=vx)
            self.assertEqual(c._navigate_step(o,edge,t).reason,'jump_brake')
        o.player=Actor(Box(665,408,695,458),.99,vx=-60)
        self.assertEqual(c._navigate_step(o,edge,1.36).reason,'jump')

    def test_continuing_coast_restarts_the_position_dwell(self):
        c,o,edge=self.case();c._navigate_step(o,edge,1)
        for t,x in [(1.1,675),(1.2,681),(1.3,687),(1.4,693)]:
            o.player=Actor(Box(x-15,408,x+15,458),.99,vx=20)
            self.assertEqual(c._navigate_step(o,edge,t).reason,'jump_brake')
            self.assertIsNone(c.transition)

    def test_braking_starts_before_fast_approach_reaches_narrow_overlap(self):
        c,o,edge=self.case(x=660,vx=150)
        d=c._navigate_step(o,edge,1)
        self.assertEqual(d.reason,'jump_brake');self.assertFalse(d.keys)
        self.assertIsNone(c.transition)

    def test_runtime_idle_recovery_does_not_interrupt_owned_jump_brake(self):
        from autofarm.realtime.recovery import ActiveRecovery
        c,o,edge=self.case();recovery=ActiveRecovery()
        for t in (1,1.3,1.6,1.9):
            o.captured_at=t
            o.player=Actor(o.player.box.moved(6,0),.99,vx=80)
            d=c._navigate_step(o,edge,t)
            self.assertEqual(d.reason,'jump_brake')
            self.assertEqual(recovery.apply(o,d,t,1366,controller=c),d)


if __name__=='__main__':unittest.main()
