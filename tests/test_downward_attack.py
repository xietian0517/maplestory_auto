import unittest
from autofarm.realtime.control import Controller
from autofarm.realtime.model import Actor,Box,Decision,MotionProfile,Observation,Platform


class DownwardAttackTests(unittest.TestCase):
    def observation(self,t=1,dy=66,dx=-30):
        p=Actor(Box(398,520,428,568),.99)
        m=Actor(Box(413+dx-25,568+dy-46,413+dx+25,568+dy),.99,track_id=7)
        return Observation(1,t,p,[m],[Platform('leaf',350,455,568),Platform('ground',0,1200,634)])

    def controller(self):
        c=Controller(MotionProfile(180,110,140,True),True);c.last_epoch=0
        c.acknowledge(Decision(frozenset({'left'})),.8)
        return c

    def test_leaf_can_attack_below_without_jump_or_melee_retreat(self):
        c=self.controller();o=self.observation();d=c.decide(o,1)
        self.assertEqual(d.keys,{'shift'});self.assertIsNone(c.jump_combat)
        self.assertEqual(c.orient_attack(o,d,1).keys,{'shift'})

    def test_visible_lower_target_precedes_unnecessary_higher_jump(self):
        c=self.controller();o=self.observation()
        o.monsters.append(Actor(Box(490,445,535,490),.99,track_id=8))
        self.assertEqual(c.decide(o,1).target,'7');self.assertIsNone(c.jump_combat)

    def test_does_not_expand_unverified_upward_or_far_downward_range(self):
        c=self.controller()
        for dy in (-65,100):
            o=self.observation(dy=dy)
            self.assertIsNone(c.fight(o,o.platforms[0],1))

    def test_same_height_contact_is_cleared_on_foot_instead_of_jumping(self):
        # 跳A 已禁用：同层近身目标必须由站立射击路径处理（先让位或直接打），
        # 不能再起跳对齐高度。
        c=self.controller();d=c.decide(self.observation(dy=0),1)
        self.assertIsNone(c.jump_combat);self.assertNotIn('alt',d.keys)
        self.assertEqual(d.reason,'attack');self.assertEqual(d.keys,{'shift'})

    def test_direct_burst_can_remain_on_visible_lower_target(self):
        c=self.controller();c.direct_attacks=True
        d=c.decide(self.observation(),1);c.acknowledge(d,1)
        d=c.decide(self.observation(1.1),1.1)
        self.assertEqual(d.reason,'direct_attack_hold');self.assertEqual(d.keys,{'shift'})


if __name__=='__main__':unittest.main()
