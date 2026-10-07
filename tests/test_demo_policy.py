import copy
import unittest
from types import SimpleNamespace
from autofarm.realtime.control import Controller
from autofarm.realtime.model import Actor,Box,MotionProfile,Observation,Platform
from autofarm.realtime.demo_policy import features,PlatformIntent


class DemoPolicyTests(unittest.TestCase):
    def row(self):
        return dict(reason='',floor_id='a',player=dict(box=[100,50,130,100],vx=0,vy=0),
            platforms=[dict(id='a',left=0,right=500,y=100),dict(id='b',left=0,right=500,y=200)],
            monsters=[dict(box=[200,155,240,200],confidence=.99)],navigation_targets=[])

    def test_relative_state_features_ignore_common_camera_translation(self):
        a=self.row();b=copy.deepcopy(a)
        for p in b['platforms']:p['left']+=30;p['right']+=30;p['y']+=20
        for actor in [b['player'],*b['monsters']]:
            actor['box']=[v+(30 if i%2==0 else 20) for i,v in enumerate(actor['box'])]
        self.assertEqual(features(a,['a','b']),features(b,['a','b']))

    def test_unknown_identity_cannot_get_a_policy_prediction(self):
        row=self.row();row['reason']='player_not_found'
        self.assertIsNone(features(row,['a','b']))

    def test_json_tree_scores_are_suggestions_and_reject_cycles(self):
        data=dict(version=1,platform_ids=['a','b'],classes=['a','b'],trees=[dict(
            left=[-1],right=[-1],feature=[-2],threshold=[-2],value=[[.25,.75]])])
        model=PlatformIntent(data)
        self.assertEqual(model.predict(self.row()),[('b',.75),('a',.25)])
        data['trees'][0].update(left=[0],right=[0],feature=[0],threshold=[.5])
        with self.assertRaises(ValueError):PlatformIntent(data)

    def route_case(self,mode,goal='c',vote=.8):
        c=Controller(MotionProfile(180,100,150,True),True);c.last_epoch=0;c.no_enemy_since=0
        c.platform_policy=SimpleNamespace(predict=lambda row:[(goal,vote)]);c.policy_mode=mode
        o=Observation(1,2,Actor(Box(285,52,315,100),.99),
            platforms=[Platform('a',0,600,100),Platform('b',0,600,200),Platform('c',0,600,300)],
            navigation_targets=[Actor(Box(310,150,350,200),.99)])
        return c,o

    def test_shadow_does_not_override_rule_route_but_active_can_suggest_reachable_goal(self):
        for mode,expected in [('shadow','b'),('active','c')]:
            c,o=self.route_case(mode);d=c.decide(o,2)
            self.assertEqual(d.target,expected);self.assertTrue(c.policy_advice['eligible'])
            o.captured_at=2.15
            self.assertEqual(c.decide(o,2.15).target,expected)
            self.assertIsNone(c.policy_advice)  # An active transfer is not replanned by the model.

    def test_unknown_goal_or_weak_vote_falls_back_to_visible_enemy_route(self):
        for goal,vote in [('missing',.9),('c',.4),('a',.9)]:
            c,o=self.route_case('active',goal,vote)
            self.assertEqual(c.decide(o,2).target,'b');self.assertFalse(c.policy_advice['eligible'])

    def test_model_cannot_outrank_current_attack_or_missing_identity(self):
        c,o=self.route_case('active');o.monsters=[Actor(Box(440,52,480,100),.99)]
        self.assertNotEqual(c.decide(o,2).target,'c');self.assertIsNone(c.policy_advice)
        o.player=None;o.reason='player_not_found';o.captured_at=2.1
        self.assertFalse(c.decide(o,2.1).keys);self.assertIsNone(c.policy_advice)

    def test_opted_in_current_floor_prediction_keeps_search_on_supported_platform(self):
        c,o=self.route_case('active','a',.8);c.policy_allow_stay=True
        d=c.decide(o,2)
        self.assertEqual(d.reason,'learned_platform_patrol');self.assertEqual(d.keys,{'left'})
        self.assertIsNone(c.transition);self.assertTrue(c.policy_advice['local_search'])

    def test_shadow_or_weak_stay_preserves_existing_route(self):
        for mode,vote in [('shadow',.8),('active',.6)]:
            c,o=self.route_case(mode,'a',vote);c.policy_allow_stay=True
            self.assertEqual(c.decide(o,2).target,'b')

    def test_current_floor_suggestion_cannot_override_forced_exploration(self):
        c,o=self.route_case('active','a',.8);c.policy_allow_stay=True
        c.explore_origin='a';c.explore_until=10
        self.assertNotEqual(c.decide(o,2).reason,'learned_platform_patrol')

    def test_local_search_survives_prediction_flicker_but_expires(self):
        c,o=self.route_case('active','a',.8);c.policy_allow_stay=True;c.policy_search_seconds=1.2
        self.assertEqual(c.decide(o,2).reason,'learned_platform_patrol')
        c.platform_policy.predict=lambda row:[('b',.9)]
        o.captured_at=2.3
        self.assertEqual(c.decide(o,2.3).reason,'learned_platform_patrol')
        o.captured_at=3.3
        self.assertEqual(c.decide(o,3.3).target,'b')

    def test_search_commit_cannot_outrank_combat_or_identity_loss(self):
        c,o=self.route_case('active','a',.8);c.policy_allow_stay=True;c.policy_search_seconds=1.2
        c.decide(o,2)
        o.monsters=[Actor(Box(440,52,480,100),.99)];o.captured_at=2.2
        self.assertNotEqual(c.decide(o,2.2).reason,'learned_platform_patrol')
        o.player=None;o.reason='player_not_found';o.captured_at=2.3
        self.assertFalse(c.decide(o,2.3).keys)
        self.assertIsNone(c.policy_search_floor)

    def test_search_commit_is_not_extended_each_frame(self):
        c,o=self.route_case('active','a',.8);c.policy_allow_stay=True;c.policy_search_seconds=1.2
        c.decide(o,2)
        o.captured_at=2.3;c.decide(o,2.3)
        self.assertEqual(c.policy_search_until,3.2)
        self.assertEqual(c.policy_search_retry,6)


if __name__=='__main__':unittest.main()
