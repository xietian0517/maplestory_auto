import unittest
from autofarm.realtime.control import Controller
from autofarm.realtime.model import Actor,Box,Decision,MotionProfile,Observation,Platform


class RangedClearanceTests(unittest.TestCase):
    def actor(self,x,y=400,ident=1,confidence=.99):
        return Actor(Box(x-25,y-45,x+25,y),confidence,track_id=ident)

    def observation(self,t=1,monsters=None):
        return Observation(1,t,Actor(Box(285,352,315,400),.99),
            monsters=monsters or [self.actor(350,ident=1),self.actor(450,ident=2)],
            platforms=[Platform('ground',0,800,400)])

    def controller(self,direct=False):
        c=Controller(MotionProfile(180,110,140,True),True);c.last_epoch=0
        c.direct_attacks=direct;c.acknowledge(Decision(frozenset({'right'})),.8)
        return c

    def test_near_enemy_blocks_distant_target_and_hybrid_makes_room_on_foot(self):
        c=self.controller();o=self.observation();d=c.decide(o,1)
        # 近身敌人仍然压过远处目标，但处理方式是同层让位而不是起跳。
        self.assertIsNone(c.jump_combat);self.assertEqual(d.reason,'retreat')
        self.assertEqual(d.keys,{'left'})

    def test_clear_opposite_direction_remains_available(self):
        c=self.controller();o=self.observation();o.monsters.append(self.actor(120,ident=3))
        d=c.decide(o,1);self.assertIsNone(c.jump_combat)
        o.captured_at=1.07;d=c.decide(o,1.07)
        self.assertEqual(d.target,'3');self.assertEqual(c.facing,'left')

    def test_direct_hold_is_cancelled_when_close_enemy_enters_firing_direction(self):
        c=self.controller(True);o=self.observation(monsters=[self.actor(450,ident=2)])
        d=c.decide(o,1);c.acknowledge(d,1)
        d=c.decide(self.observation(1.1),1.1)
        self.assertNotIn('shift',d.keys);self.assertEqual(d.reason,'ranged_make_room')

    def test_lower_target_and_low_confidence_blocker_do_not_force_a_jump(self):
        for monsters in ([self.actor(330,466)],
                         [self.actor(350,confidence=.5),self.actor(450,ident=2)]):
            c=self.controller();d=c.decide(self.observation(monsters=monsters),1)
            self.assertEqual(d.keys,{'shift'});self.assertIsNone(c.jump_combat)

    def test_actual_round046_two_monster_geometry(self):
        c=self.controller();o=self.observation()
        o.player=Actor(Box(597,584,627,632),.99)
        o.platforms=[Platform('ground',0,1366,638)]
        o.monsters=[Actor(Box(643,582,693,632),.99,track_id=60),
                    Actor(Box(698,591,748,641),.99,track_id=61)]
        d=c.decide(o,1)
        # 56px 已进入近战判定，站立策略先向左让出射击距离，而不是起跳对齐。
        self.assertIsNone(c.jump_combat);self.assertEqual(d.reason,'retreat')
        self.assertEqual(d.keys,{'left'})

    def test_safe_distance_retreats_before_true_melee_range(self):
        # 70px 还没到 65px 近战判定，但已经近到会顶掉飞镖，必须提前让位。
        c=self.controller();d=c.decide(self.observation(monsters=[self.actor(370)]),1)
        self.assertNotIn('shift',d.keys)
        self.assertEqual(d.reason,'retreat');self.assertEqual(d.keys,{'left'})

    def test_shot_beyond_the_safe_distance_is_still_allowed(self):
        c=self.controller();d=c.decide(self.observation(monsters=[self.actor(420)]),1)
        self.assertIn('shift',d.keys)

    def test_blocker_in_firing_direction_cancels_a_distant_target(self):
        c=self.controller();p=Box(285,352,315,400)
        target=self.actor(600,ident=2)
        self.assertFalse(c.clear_ranged_target(target,p,[target,self.actor(330,ident=3)]))
        self.assertTrue(c.clear_ranged_target(target,p,[target,self.actor(500,ident=3,confidence=.5)]))

    def test_engaged_burst_stops_when_a_monster_enters_the_safe_distance(self):
        c=Controller(MotionProfile(180,110,140,True),True);c.last_epoch=0
        c.sustain_farming=True;c.applied_facing=c.facing='right'
        o=self.observation(monsters=[self.actor(450,ident=2)])
        d=c.decide(o,1);self.assertIn('shift',d.keys);c.acknowledge(d,1)
        d=c.decide(self.observation(t=1.1,monsters=[self.actor(340,ident=2)]),1.1)
        self.assertNotIn('shift',d.keys);self.assertEqual(d.reason,'retreat')


if __name__=='__main__':unittest.main()
