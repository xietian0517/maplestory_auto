import unittest
from autofarm.realtime.control import Controller
from autofarm.realtime.firing_position import choose_position,clear_intervals
from autofarm.realtime.model import Actor,Box,Decision,MotionProfile,Observation,Platform
from autofarm.realtime.recovery import ActiveRecovery


class FiringPositionTests(unittest.TestCase):
    def test_distant_monster_does_not_disqualify_entire_platform(self):
        p=Platform('wide',0,800,300);m=MotionProfile(180,110,200,True)
        spans=clear_intervals(p,[Actor(Box(85,255,115,300),.99)],m,(1366,694))
        self.assertEqual(spans,[(185,800)])

    def test_partly_visible_and_top_platform_keep_visible_standing_interval(self):
        self.assertEqual(clear_intervals(Platform('top',-100,300,35),[],MotionProfile(),(1366,694)),[(0,300)])

    def test_navigation_launch_uses_selected_landing_interval(self):
        c=Controller(MotionProfile(180,110,200,True),navigate=True)
        c.sustain_farming=True;c.firing_goal='ledge';c.firing_bounds={'ledge':(400,550)}
        o=Observation(1,1,Actor(Box(85,352,115,400),.99),
            platforms=[Platform('ground',0,600,400),Platform('ledge',0,600,330)])
        d=c._navigate_step(o,('ground','ledge','jump'),1)
        self.assertIn('right',d.keys);self.assertNotIn('alt',d.keys)

    def test_clear_interval_edge_is_not_used_as_walk_off_ledge(self):
        c=Controller(MotionProfile(180,110,200,True),navigate=True)
        c.sustain_farming=True;c.firing_goal='lower';c.firing_bounds={'upper':(200,400)}
        c.failure_counts[('upper','lower')]=1
        o=Observation(1,1,Actor(Box(285,252,315,300),.99),
            platforms=[Platform('upper',0,600,300),Platform('lower',100,500,400)])
        self.assertNotEqual(c._navigate_step(o,('upper','lower','drop'),1).reason,'walk_off_drop')

    def scene(self,t=1):
        return Observation(1,t,Actor(Box(85,352,115,400),.99),
            [Actor(Box(x-15,355,x+15,400),.99,track_id=i) for i,x in enumerate([310,360,410],1)],
            [Platform('ground',0,800,400),Platform('leaf',150,250,330)])

    def test_chooses_empty_firing_ledge_not_monster_floor(self):
        o=self.scene();m=MotionProfile(180,110,200,True)
        choice=choose_position(o,o.platforms[0],m)
        self.assertEqual(choice[1],'leaf');self.assertEqual(choice[4],3)
        self.assertEqual(choice[3][-1][1],'leaf')

    def test_occupied_unreachable_and_offscreen_ledge_rejected(self):
        m=MotionProfile(180,110,200,True)
        o=self.scene();o.monsters.append(Actor(Box(185,285,215,330),.99,track_id=9))
        self.assertNotEqual(choose_position(o,o.platforms[0],m)[1],'leaf')
        o=self.scene();self.assertNotEqual(choose_position(o,o.platforms[0],m,{('ground','leaf')})[1],'leaf')
        self.assertNotEqual(choose_position(o,o.platforms[0],m,viewport=(200,694))[1],'leaf')

    def test_navigation_and_combat_detections_are_not_double_counted(self):
        o=self.scene();o.navigation_targets=o.monsters[:]
        self.assertEqual(choose_position(o,o.platforms[0],MotionProfile(180,110,200,True))[4],3)

    def test_counts_targets_on_both_sides_of_empty_platform(self):
        o=self.scene();o.monsters=[Actor(Box(x-15,355,x+15,400),.99,track_id=i)
                                  for i,x in enumerate([100,310],1)]
        choice=choose_position(o,o.platforms[0],MotionProfile(180,110,200,True))
        self.assertEqual(choice[1],'leaf');self.assertEqual(choice[4],2)

    def armed(self):
        c=Controller();c.sustain_farming=True;c.last_epoch=0;c.applied_facing=c.facing='right'
        o=self.scene();d=c.decide(o,1);c.acknowledge(d,1)
        return c

    def test_visible_target_does_not_trigger_three_second_walk(self):
        c=self.armed()
        for i in range(1,51):
            t=1+i*.1;d=c.decide(self.scene(t),t)
            self.assertEqual(d.keys,{'shift'});c.acknowledge(d,t)

    def test_new_firing_destination_does_not_abandon_current_attack(self):
        c=self.armed();c.navigate=True;c.motion=MotionProfile(180,110,200,True)
        self.assertEqual(c.decide(self.scene(1.1),1.1).keys,{'shift'})

    def test_short_direction_refresh_retains_existing_engagement(self):
        c=self.armed();o=self.scene(1.1)
        c.fight(o,o.platforms[0],1.1) # New target observation while turn is submitted.
        c.acknowledge(Decision(frozenset({'right'}),'attack_face_confirm','1'),1.1)
        self.assertTrue(c.wait_engagement(self.scene(1.2),1.2))

    def test_missing_target_pauses_without_claiming_kill_or_walking(self):
        c=self.armed();o=self.scene(1.3);o.monsters=[]
        d=c.decide(o,1.3);self.assertEqual(d.reason,'engagement_reacquire_wait');self.assertFalse(d.keys)
        o.captured_at=1.7;self.assertFalse(c.wait_engagement(o,1.7))

    def repositioning_scene(self,visible=True):
        """A better firing ledge exists, but the current view also has a target."""
        monsters=[Actor(Box(cx-15,292,cx+15,340),.99,track_id=i)
                  for i,cx in enumerate((400,450,500),1)]
        if visible: monsters.append(Actor(Box(295,365,325,400),.99,track_id=9))
        o=Observation(1,0,Actor(Box(85,352,115,400),.99),monsters,
                      [Platform('ground',0,800,400),Platform('leaf',150,250,330)])
        o.captured_at=0
        return o

    def test_visible_target_suppresses_multi_floor_reposition(self):
        o=self.repositioning_scene();m=MotionProfile(180,110,200,True)
        choice=choose_position(o,o.platforms[0],m)
        self.assertEqual(choice[1],'leaf');self.assertTrue(choice[3])
        c=Controller(m,navigate=True);c.sustain_farming=True
        for t in (0,.1,.2,.3):
            o.captured_at=t;d=c.decide(o,t);c.acknowledge(d,t)
            self.assertNotIn(d.reason,('approach_launch','jump','drop','walk_off_drop'))
            self.assertIsNone(c.pending_edge);self.assertIsNone(c.transition);self.assertIsNone(c.rope_climber)
        self.assertIn('shift',d.keys)

    def test_without_a_visible_target_the_same_floor_change_is_taken(self):
        o=self.repositioning_scene(visible=False);m=MotionProfile(180,110,200,True)
        c=Controller(m,navigate=True);c.sustain_farming=True
        d=c.decide(o,0)
        self.assertEqual(d.reason,'approach_launch');self.assertEqual(c.pending_edge,('ground','leaf','jump'))

    def test_recovery_does_not_walk_into_monsters_when_no_firing_route(self):
        c=Controller();c.sustain_farming=True;o=self.scene();r=ActiveRecovery()
        d=Decision(reason='firing_position_wait')
        for t in (1,2,8):
            o.captured_at=t;self.assertEqual(r.apply(o,d,t,1366,controller=c),d)
