import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest

from autofarm.realtime.farm_plan import FarmPlanExecutor
from autofarm.realtime.pipeline import DecisionPipeline
from autofarm.realtime.pipeline_replay import replay_trace
from autofarm.realtime.policy import PolicyConfig
from tests.test_policy_pipeline import base, observed, intent, ManualPool


def farm(t=1, duration=60, x=300, target='a', direction='right'):
    return intent('farm', target=target, x=x, direction=direction, t=t, duration=duration)


class FarmPlanTests(unittest.TestCase):
    def test_suspended_rope_requires_unique_geometry_and_observed_progress(self):
        from autofarm.realtime.model import Rope
        from autofarm.realtime.farm_plan import suspended_rope_origin
        b=base();e=FarmPlanExecutor(b,True)
        o=observed(monsters=False);o.player=replace(o.player,box=replace(o.player.box,y1=102,y2=150))
        o.ropes=[Rope(300,100,200)]
        self.assertIsNotNone(suspended_rope_origin(o))
        self.assertTrue(e.accept(farm(),o,1,scene_id='scene'))
        self.assertEqual(e.phase,'rope_resume');self.assertTrue(e.busy)
        self.assertEqual(e.step(o,1).keys,{'up'})
        o.captured_at=1.7
        self.assertFalse(e.step(o,1.7).keys);self.assertIsNone(e.current)
        self.assertEqual(e.outcomes[-1]['reason'],'rope_resume_unconfirmed')
        o.ropes.append(Rope(304,100,200))
        self.assertIsNone(suspended_rope_origin(o))

    def test_rope_resume_reaches_ai_goal_then_fires_without_new_decision(self):
        from autofarm.realtime.model import Rope
        b=base();b.applied_facing='right';b.applied_face_at=0;e=FarmPlanExecutor(b,True)
        o=observed(monsters=False);o.player=replace(o.player,box=replace(o.player.box,y1=102,y2=150))
        o.ropes=[Rope(300,100,200)];e.accept(farm(),o,1,scene_id='scene');e.step(o,1)
        o.player=replace(o.player,box=replace(o.player.box,y1=82,y2=130));o.captured_at=1.2
        self.assertEqual(e.step(o,1.2).keys,{'up'})
        o.player=replace(o.player,box=replace(o.player.box,y1=52,y2=100));o.captured_at=1.4;e.step(o,1.4)
        o.captured_at=1.5;e.step(o,1.5)
        o.captured_at=1.7;self.assertEqual(e.step(o,1.7).keys,{'shift'})
        self.assertEqual(e.current.target,'a')

    def test_requires_explicit_complete_goal_and_opt_in(self):
        with self.assertRaises(ValueError): farm(direction='')
        with self.assertRaises(ValueError): farm(x=None)
        with self.assertRaises(ValueError): farm(duration=121)
        from autofarm.realtime.actions import ActionExecutor
        self.assertFalse(ActionExecutor(base()).accept(farm(), observed(), 1, scene_id='scene'))
        with self.assertRaises(ValueError):PolicyConfig.parse(dict(version=1,hierarchical=1))

    def test_rejects_transit_only_destination_and_bad_stand(self):
        b=base();b.firing_platforms={'a'};e=FarmPlanExecutor(b,True)
        self.assertFalse(e.accept(farm(target='b'),observed(),1,scene_id='scene'))
        self.assertEqual(e.results[-1]['reason'],'intent_farm_destination_not_firing_support')
        self.assertFalse(e.accept(farm(x=699),observed(),1,scene_id='scene'))

    def test_arrival_positions_and_fires_without_another_reply(self):
        b=base();b.applied_facing='right';b.applied_face_at=0;e=FarmPlanExecutor(b,True)
        self.assertTrue(e.accept(farm(target='b'),observed(),1,scene_id='scene'))
        self.assertTrue(e.busy)
        o=observed(t=1.3);o.player=replace(o.player,box=replace(o.player.box,y1=172,y2=220))
        e.step(o,1.3);o.captured_at=1.5;e.step(o,1.5)
        o.captured_at=1.7;d=e.step(o,1.7)
        self.assertEqual(d.keys,{'shift'});self.assertEqual(e.phase,'fire')
        self.assertEqual(e.current.target,'b')

    def test_continuous_fire_across_leases_and_slow_review(self):
        b=base();b.applied_facing='right';b.applied_face_at=0;e=FarmPlanExecutor(b)
        self.assertTrue(e.accept(farm(duration=90),observed(),1,scene_id='scene'))
        for n in range(1000):
            t=1+n*.05;o=observed(t=t,monsters=False)
            d=e.step(o,t);b.acknowledge(d,t);e.acknowledge(d,True,t,o)
            if n>5:self.assertEqual(d.keys,{'shift'})
        self.assertGreater(e.attack_seconds,49)
        self.assertTrue(e.review_due());e.review_sent=True
        self.assertFalse(e.review_due());self.assertIsNotNone(e.current)
        self.assertEqual(e.current.world_x,300)

    def test_map_guard_cancel_and_actual_input_budget(self):
        b=base();b.applied_facing='right';b.applied_face_at=0;e=FarmPlanExecutor(b)
        e.accept(farm(),observed(),1,scene_id='scene')
        for n in range(30):
            t=1+n*.05;o=observed(t=t);d=e.step(o,t);e.acknowledge(d,False,t,o)
        self.assertEqual(e.attack_seconds,0)
        o=observed(t=3);o.map_epoch=1
        self.assertFalse(e.step(o,3).keys);self.assertIsNone(e.current)
        self.assertIsNone(e.motor.current)

    def test_plan_reward_unknown_after_hud_gap_and_failure_retained(self):
        e=FarmPlanExecutor(base());row=dict(exp=100,level=47,t=1)
        e.observe_feedback(dict(confirmed_experience=row))
        e.accept(farm(),observed(),1,scene_id='scene')
        e.observe_feedback(dict(confirmed_experience=dict(exp=120,level=47,t=1.4)))
        self.assertEqual(e.state()['net_exp'],20)
        e.observe_feedback(dict(confirmed_experience=None))
        e.finish('failed','chosen_plan_support_changed',2,observed(t=2))
        self.assertIsNone(e.outcomes[-1]['net_exp'])
        self.assertEqual(e.outcomes[-1]['platform'],'a')

    def test_hierarchical_trace_replays_phase_handover(self):
        with tempfile.TemporaryDirectory() as folder:
            b=base();b.applied_facing='right';b.applied_face_at=0
            # Facing is acquired through recorded input so replay starts in the same state.
            b.applied_facing='';b.applied_face_at=0
            p=DecisionPipeline(b,PolicyConfig(mode='active',hierarchical=True),folder=folder,replay=True)
            for n in range(30):
                t=1+n*.05;o=observed(t=t)
                d=p.step(o,t,scene_id='scene',delivery=farm() if n==0 else None)
                p.commit(d,True,t,o,held_keys=d.keys)
            p.close(2.5)
            result=replay_trace(folder)
            self.assertTrue(result['identical'],result)

    def test_requests_wait_for_actual_attack_and_pending_review_keeps_firing(self):
        from autofarm.realtime.actions import ActionIntent
        b=base();b.applied_facing='right';b.applied_face_at=0
        p=DecisionPipeline(b,PolicyConfig(mode='active',hierarchical=True,response_ttl=30),
                           policy=SimpleNamespace(propose=lambda r:None))
        p.runner.pool.shutdown(wait=False);pool=ManualPool();p.runner.pool=pool
        o=observed();p.step(o,1,scene_id='scene');p.commit(p.step(o,1,scene_id='scene'),True,1,o)
        req=p.runner.request
        chosen=replace(farm(duration=90),action_id=req.request_id,request_id=req.request_id,
                       issued_at=req.issued_at,valid_until=req.valid_until)
        pool.calls[0][0].set_result(chosen)
        for n in range(1,721):
            t=1+n*.05;o=observed(t=t,monsters=False)
            d=p.step(o,t,scene_id='scene');p.commit(d,True,t,o,held_keys=d.keys)
            if 1.4<t<26:self.assertEqual(len(pool.calls),1)
            if t>27:self.assertEqual(d.keys,{'shift'})
        self.assertEqual(len(pool.calls),2)
        self.assertTrue(p.status()['pending_model'])
        self.assertGreater(p.executor.attack_seconds,35)
        p.close(37)


if __name__=='__main__':unittest.main()
