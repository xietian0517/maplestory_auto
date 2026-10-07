import copy
from concurrent.futures import Future
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from autofarm.realtime.actions import ActionExecutor, ActionIntent
from autofarm.realtime.control import Controller
from autofarm.realtime.feedback import FeedbackTracker, StateHistory, observation_data, observation_from_data
from autofarm.realtime.model import Actor, Box, Decision, MotionProfile, Observation, Platform
from autofarm.realtime.pipeline import DecisionPipeline, controller_configuration, restore_controller
from autofarm.realtime.pipeline_replay import replay_trace
from autofarm.realtime.policy import (AsyncPolicy, OpenAIActionPolicy, PolicyConfig, PolicyRequest,
                                     RulePolicy, action_schema, load_policy_config)


def observed(t=1, frame=1, monsters=True):
    return Observation(frame,t,Actor(Box(285,52,315,100),.99),
        [Actor(Box(440,52,480,100),.99,track_id=7),Actor(Box(100,52,140,100),.99,track_id=8)] if monsters else [],
        [Platform('a',0,700,100),Platform('b',0,700,220)],position_quantum=16)


def intent(action='wait', target='', x=None, direction='', t=1, frame=1, ident='r', duration=.5):
    return ActionIntent(ident,'ai',ident,frame,0,'scene',t,t+15,action,target,x,direction,duration,'test choice')


def base():
    b=Controller(MotionProfile(180,100,150,True),True)
    b.safe_platforms={'a','b'};b.firing_platforms={'a','b'};b.sustain_farming=True
    return b


class ManualPool:
    def __init__(self):self.calls=[];self.closed=False
    def submit(self,fn,*args):
        future=Future();future.set_running_or_notify_cancel();self.calls.append((future,fn,args));return future
    def shutdown(self,**kw):self.closed=True


class IntentTests(unittest.TestCase):
    def test_reply_cannot_forge_observation_or_add_fields(self):
        req=PolicyRequest('r','scene',3,0,1,10,{})
        reply=dict(request_id='r',frame_id=3,map_epoch=0,action='attack',target='7',world_x=None,
                   direction='',duration=.7,reason='attack')
        self.assertEqual(ActionIntent.from_reply(reply,req).scene_id,'scene')
        for change in ({'frame_id':4},{'map_epoch':True},{'duration':float('nan')},{'action':'shell'},
                       {'world_x':float('inf')},{'secret':'extra'}):
            with self.subTest(change=change),self.assertRaises(ValueError):
                ActionIntent.from_reply({**reply,**change},req)

    def test_config_rejects_unknowns_and_invalid_budgets(self):
        for d in ({'version':1,'key':'private'},{'version':1,'interval':float('nan')},
                  {'version':1,'timeout':30,'response_ttl':1},{'version':1,'allow_transfers':1}):
            with self.subTest(d=d),self.assertRaises(ValueError):PolicyConfig.parse(d)
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(load_policy_config(d).mode,'rule')
            self.assertEqual(load_policy_config(d,'shadow').mode,'shadow')


class ExecutorTests(unittest.TestCase):
    def test_occupied_unplanned_start_floor_remains_available_for_escape(self):
        b=base();b.safe_platforms=b.firing_platforms={'b'}
        o=observed();o.monsters=[Actor(Box(285,52,315,100),.99,track_id=7)]
        e=ActionExecutor(b,True)
        self.assertIn('a',[p.id for p in e.transfer_platforms(o)])
        self.assertTrue(e.accept(intent('transfer','b'),o,1,scene_id='scene'))

    def test_short_turn_uses_turn_margin_not_longer_move_margin(self):
        b=base();b.applied_facing='left';b.applied_face_at=0;e=ActionExecutor(b)
        o=observed(monsters=False);o.player=Actor(Box(655,52,685,100),.99)
        self.assertLess(e.turn_clearance(o),30);self.assertGreater(e.clearance(o),30)
        self.assertTrue(e.accept(intent('fire',direction='right'),o,1,scene_id='scene'))
        self.assertEqual(e.step(o,1).keys,{'right'})

    def test_combat_transit_is_opt_in_and_keeps_fire_perch_restrictions(self):
        b=base();b.safe_platforms=b.firing_platforms={'a','c'};b.combat_platforms={'b'}
        o=observed(monsters=False);o.player=Actor(Box(45,52,75,100),.99)
        o.platforms=[Platform('a',0,100,100),Platform('b',100,700,160),Platform('c',550,700,100)]
        o.monsters=[Actor(Box(600,112,640,160),.99,track_id=7)]
        default=ActionExecutor(b,True)
        self.assertNotIn('b',[p.id for p in default.transfer_platforms(o)])
        self.assertFalse(default.accept(intent('transfer','c'),o,1,scene_id='scene'))
        e=ActionExecutor(b,True,True)
        self.assertIn('b',[p.id for p in e.transfer_platforms(o)])
        self.assertTrue(e.accept(intent('transfer','c'),o,1,scene_id='scene'))
        self.assertNotEqual(e.step(o,1).reason,'safe_platform_route_wait')
        e.cancel(1.1,'test');o.player=Actor(Box(285,112,315,160),.99);o.captured_at=1.2
        self.assertFalse(e.accept(intent('fire',direction='right',t=1.2),o,1.2,scene_id='scene'))
        self.assertEqual(e.results[-1]['reason'],'intent_outside_task_platforms')

    def test_long_ai_fire_remains_bounded_and_cancels_on_invalid_observation(self):
        b=base();b.applied_facing='left';b.applied_face_at=0;e=ActionExecutor(b)
        o=observed(monsters=False);chosen=intent('fire',direction='left',duration=20)
        self.assertTrue(e.accept(chosen,o,1,scene_id='scene'));d=e.step(o,1);e.acknowledge(d,True,1,o)
        o.captured_at=16;self.assertEqual(e.step(o,16).keys,{'shift'})
        o.reason='focus_lost';o.captured_at=16.1;self.assertFalse(e.step(o,16.1).keys)
        self.assertIsNone(e.current)
        with self.assertRaises(ValueError):intent('fire',direction='left',duration=20.1)
        with self.assertRaises(ValueError):intent('move',x=400,duration=20)

    def test_ai_fire_does_not_require_or_fabricate_a_monster_identity(self):
        b=base();b.applied_facing='left';b.applied_face_at=0;e=ActionExecutor(b)
        o=observed(monsters=False);chosen=intent('fire',direction='left',duration=3)
        self.assertTrue(e.accept(chosen,o,1,scene_id='scene'))
        d=e.step(o,1);self.assertEqual(d.keys,{'shift'});self.assertEqual(d.target,'')
        e.acknowledge(d,True,1,o);o.captured_at=2
        self.assertEqual(e.step(o,2).keys,{'shift'})
        o.captured_at=4.1;self.assertFalse(e.step(o,4.1).keys)
        self.assertIsNone(e.current);self.assertEqual(e.results[-1]['reason'],'fire_interval_submitted_hit_unknown')

    def test_fire_rejects_close_blocker_and_cancels_if_task_support_changes(self):
        b=base();b.applied_facing='right';b.applied_face_at=0;e=ActionExecutor(b);o=observed()
        o.monsters=[Actor(Box(310,52,350,100),.99,track_id=7)]
        self.assertFalse(e.accept(intent('fire',direction='right'),o,1,scene_id='scene'))
        o.monsters=[];self.assertTrue(e.accept(intent('fire',direction='right'),o,1,scene_id='scene'))
        b.firing_platforms={'b'};o.captured_at=1.1
        self.assertFalse(e.step(o,1.1).keys);self.assertIsNone(e.current)

    def test_fire_requires_direction_only_and_obeys_turn_clearance(self):
        with self.assertRaises(ValueError):intent('fire')
        with self.assertRaises(ValueError):intent('fire',target='7',direction='right')
        b=base();e=ActionExecutor(b);o=observed(monsters=False)
        o.player=Actor(Box(675,52,705,100),.99)
        self.assertTrue(e.accept(intent('fire',direction='right'),o,1,scene_id='scene'))
        self.assertFalse(e.step(o,1).keys);self.assertIsNone(e.current)

    def test_fire_trace_replays_without_model_or_sprite_targets(self):
        with tempfile.TemporaryDirectory() as folder:
            p=DecisionPipeline(base(),PolicyConfig(mode='active'),replay=True,folder=folder)
            for index,t in enumerate([1,1.1,2,4.2]):
                o=observed(t,index,monsters=False)
                d=p.step(o,t,scene_id='scene',delivery=intent('fire',direction='right',duration=3) if index==0 else None)
                p.commit(d,True,t,o,d.keys)
            p.close(4.3);self.assertTrue(replay_trace(folder)['identical'])

    def test_ai_chosen_lane_survives_id_change_but_does_not_switch_direction(self):
        b=base();b.applied_facing='right';b.applied_face_at=0
        p=DecisionPipeline(b,PolicyConfig(mode='active'),replay=True);o=observed()
        d=p.step(o,1,scene_id='scene',delivery=intent('attack','lane:right',duration=3))
        self.assertEqual(d.keys,{'shift'});self.assertEqual(d.target,'lane:right');p.commit(d,True,1,o,{'shift'})
        o=observed(1.1,2);o.monsters=[replace(o.monsters[0],track_id=70)]
        d=p.step(o,1.1,scene_id='scene');self.assertEqual(d.keys,{'shift'})
        p.commit(d,True,1.1,o,{'shift'})
        o=observed(1.2,3);o.monsters=[o.monsters[1]]  # Only opposite lane visible.
        d=p.step(o,1.2,scene_id='scene');self.assertFalse(d.keys)
        self.assertIsNone(p.executor.current)
        self.assertEqual(p.last_snapshot['attack_lanes'][0]['current_members'],['8'])
        self.assertEqual(p.cycle_events[-1]['reason'],'chosen_attack_lane_empty')

    def test_lane_cannot_attack_unseen_unsafe_or_close_targets(self):
        for change in ('none','near','support','wrong_direction','unknown_lane'):
            with self.subTest(change=change):
                b=base();b.applied_facing='right';b.applied_face_at=0;e=ActionExecutor(b);o=observed()
                chosen=intent('attack','lane:right')
                if change=='none':o.monsters=[]
                elif change=='near':o.monsters=[Actor(Box(310,52,350,100),.99,track_id=7)]
                elif change=='support':b.firing_platforms={'b'}
                elif change=='wrong_direction':chosen=replace(chosen,direction='left')
                else:chosen=replace(chosen,target='lane:up')
                self.assertFalse(e.accept(chosen,o,1,scene_id='scene'))

    def test_lane_turn_requires_space_and_accepted_direction_before_attack(self):
        b=base();e=ActionExecutor(b);o=observed()
        self.assertTrue(e.accept(intent('attack','lane:right',duration=3),o,1,scene_id='scene'))
        d=e.step(o,1);self.assertEqual(d.keys,{'right'});self.assertIsNone(e.applied_at)
        b.acknowledge(d,1);e.acknowledge(d,True,1,o)
        o.captured_at=1.03;self.assertEqual(e.step(o,1.03).keys,{'right'})
        o.captured_at=1.1;self.assertEqual(e.step(o,1.1).keys,{'shift'})
        e.acknowledge(e.step(o,1.1),True,1.1,o);self.assertEqual(e.applied_at,1.1)

    def test_lane_interval_and_failure_history_replay_identically(self):
        with tempfile.TemporaryDirectory() as folder:
            b=base()
            p=DecisionPipeline(b,PolicyConfig(mode='active'),replay=True,folder=folder)
            o=observed();d=p.step(o,1,scene_id='scene',delivery=intent('attack','lane:right',duration=3))
            p.commit(d,True,1,o,d.keys)
            o=observed(1.1,2);d=p.step(o,1.1,scene_id='scene');p.commit(d,True,1.1,o,d.keys)
            o=observed(1.5,3);o.monsters=[replace(o.monsters[0],track_id=99)]
            d=p.step(o,1.5,scene_id='scene');p.commit(d,True,1.5,o,d.keys)
            o=observed(2,4,monsters=False);d=p.step(o,2,scene_id='scene');p.commit(d,True,2,o,d.keys)
            o=observed(2.1,5);d=p.step(o,2.1,scene_id='scene');p.commit(d,True,2.1,o,d.keys)
            self.assertTrue(any(e['reason']=='chosen_attack_lane_empty' for e in p.last_snapshot['action_results']))
            p.close(2.2);self.assertTrue(replay_trace(folder)['identical'])

    def test_wait_outranks_visible_enemy_and_current_preset_rules(self):
        p=DecisionPipeline(base(),PolicyConfig(mode='active'),replay=True)
        o=observed();d=p.step(o,1,scene_id='scene',delivery=intent())
        self.assertFalse(d.keys);self.assertEqual(d.reason,'policy_wait')
        p.commit(d,True,1,o)
        self.assertEqual(p.status()['counts']['accepted'],1)
        self.assertEqual(p.last_trace['source'],'ai_executor')

    def test_ai_can_choose_non_nearest_target_in_sustain_farming(self):
        b=base();b.applied_facing='left';b.applied_face_at=0
        p=DecisionPipeline(b,PolicyConfig(mode='active'),replay=True)
        o=observed();d=p.step(o,1,scene_id='scene',delivery=intent('attack','8'))
        self.assertEqual(d.keys,{'shift'});self.assertEqual(d.target,'8')
        self.assertIsNone(b.policy_advice)

    def test_attack_turn_does_not_start_attack_duration(self):
        e=ActionExecutor(base());o=observed()
        self.assertTrue(e.accept(intent('attack','7'),o,1,scene_id='scene'))
        d=e.step(o,1);self.assertEqual(d.keys,{'right'})
        e.base.acknowledge(d,1);e.acknowledge(d,True,1,o);self.assertIsNone(e.applied_at)
        o.captured_at=1.1;d=e.step(o,1.1);self.assertEqual(d.keys,{'shift'})
        e.acknowledge(d,True,1.1,o);self.assertEqual(e.applied_at,1.1)

    def test_move_uses_world_coordinates_and_confirms_stopping(self):
        e=ActionExecutor(base());o=observed(monsters=False)
        i=intent('move',x=400);self.assertTrue(e.accept(i,o,1,offset=(40,0),scene_id='scene'))
        self.assertEqual(e.step(o,1,(40,0)).keys,{'right'})
        o.player=Actor(Box(425,52,455,100),.99);o.captured_at=1.2
        self.assertFalse(e.step(o,1.2,(40,0)).keys)
        o.captured_at=1.4;e.step(o,1.4,(40,0))
        self.assertIsNone(e.current);self.assertEqual(e.results[-1]['reason'],'position_observed')

    def test_target_crossing_rechecks_facing_and_disappearance_is_not_kill(self):
        b=base();b.applied_facing='right';b.applied_face_at=0;e=ActionExecutor(b);o=observed()
        e.accept(intent('attack','7'),o,1,scene_id='scene')
        self.assertEqual(e.step(o,1).keys,{'shift'})
        o.monsters=[Actor(Box(100,52,140,100),.99,track_id=7)];o.captured_at=1.1
        self.assertEqual(e.step(o,1.1).keys,{'left'})
        o.monsters=[];o.captured_at=1.2;self.assertFalse(e.step(o,1.2).keys)
        self.assertEqual(e.results[-1]['reason'],'target_lost_not_a_kill')

    def test_expired_wrong_scene_future_and_unsupported_movement_are_rejected(self):
        e=ActionExecutor(base());o=observed()
        for i,now in ((replace(intent(),valid_until=1.01),1.02),
                      (replace(intent(),scene_id='other'),1),
                      (replace(intent(),frame_id=10),1),(intent('move',x=699),1),
                      (intent('transfer','b'),1)):
            with self.subTest(i=i):self.assertFalse(e.accept(i,o,now,scene_id='scene'))

    def test_missing_identity_cancels_action_even_after_submission(self):
        p=DecisionPipeline(base(),PolicyConfig(mode='active'),replay=True);o=observed()
        d=p.step(o,1,scene_id='scene',delivery=intent('move',x=450));p.commit(d,True,1,o,{'right'})
        o.player=None;o.reason='player_not_found';o.captured_at=1.1
        d=p.step(o,1.1,scene_id='scene');self.assertFalse(d.keys)
        self.assertIsNone(p.executor.current);self.assertFalse(p.history.rows)

    def test_shadow_does_not_execute_wait_advice(self):
        b=base();b.applied_facing='right';b.applied_face_at=0
        p=DecisionPipeline(b,PolicyConfig(mode='shadow'),replay=True)
        d=p.step(observed(),1,scene_id='scene',delivery=intent())
        self.assertNotEqual(d.reason,'policy_wait');self.assertIsNone(p.executor.current)
        self.assertTrue(any(e['event']=='shadow_advice' for e in p.cycle_events))

    def test_transfer_reuses_primitive_but_never_rule_goal_selection(self):
        b=base();e=ActionExecutor(b,True);o=observed(monsters=False)
        b.decide=lambda *args: self.fail('rule selection called by AI executor')
        self.assertTrue(e.accept(intent('transfer','b'),o,1,scene_id='scene'))
        d=e.step(o,1);self.assertTrue(d.keys.intersection({'down','alt'}))
        self.assertTrue(e.busy)
        self.assertFalse(e.accept(intent(ident='later'),o,1,scene_id='scene'))
        o.player=Actor(Box(285,172,315,220),.99);o.captured_at=1.3;e.step(o,1.3)
        o.captured_at=1.6;e.step(o,1.6)
        o.captured_at=1.9;e.step(o,1.9)
        self.assertIsNone(e.current);self.assertIn(('a','b'),b.verified_edges)

    def test_attack_minimum_commit_cannot_be_superseded(self):
        b=base();b.applied_facing='right';b.applied_face_at=0;e=ActionExecutor(b);o=observed()
        e.accept(intent('attack','7',duration=.05),o,1,scene_id='scene')
        d=e.step(o,1);e.acknowledge(d,True,1,o)
        o.captured_at=1.1
        self.assertFalse(e.accept(intent(ident='next'),o,1.1,scene_id='scene'))
        self.assertEqual(e.step(o,1.1).keys,{'shift'})


class AsyncTests(unittest.TestCase):
    def test_async_pipeline_archives_current_image_and_active_wait_replays(self):
        class WaitPolicy:
            def propose(self,r):
                return ActionIntent.from_reply(dict(request_id=r.request_id,frame_id=r.frame_id,map_epoch=r.map_epoch,
                    action='wait',target='',world_x=None,direction='',duration=.5,reason='fixture wait'),r)
        with tempfile.TemporaryDirectory() as root:
            p=DecisionPipeline(base(),PolicyConfig(mode='active'),folder=root,policy=WaitPolicy())
            p.runner.pool.shutdown(wait=False);pool=ManualPool();p.runner.pool=pool
            image=np.zeros((120,700,3),np.uint8)
            o=observed();d=p.step(o,1,image,scene_id='scene');p.commit(d,False,1,o,simulated=True)
            future,fn,args=pool.calls[0];future.set_result(fn(*args))
            o.captured_at=1.1;o.frame_id=2;image[:]=255
            d=p.step(o,1.1,image,scene_id='scene');p.commit(d,False,1.1,o,simulated=True)
            self.assertEqual(d.reason,'policy_wait');self.assertEqual(p.counts['accepted'],1)
            files=list((Path(root)/'policy_requests').glob('*/request.json'));self.assertEqual(len(files),1)
            archived=json.loads(files[0].read_text(encoding='utf-8'))
            self.assertEqual(archived['state']['image_observed_at'],[1])
            self.assertTrue((files[0].parent/'0.png').exists())
            p.close(1.2);self.assertTrue(replay_trace(root)['identical'])

    def test_one_inflight_latest_state_and_expired_delivery(self):
        pool=ManualPool();runner=AsyncPolicy(SimpleNamespace(propose=lambda r:None),PolicyConfig(mode='active'),pool)
        o=observed();self.assertTrue(runner.submit(o,1,'scene',{}))
        self.assertFalse(runner.submit(o,2,'scene',{}));self.assertEqual(len(pool.calls),1)
        request=runner.request
        pool.calls[0][0].set_result(intent(t=1,ident=request.request_id))
        self.assertIsNone(runner.poll(20))
        self.assertEqual(runner.events[-1]['event'],'policy_response_rejected')
        runner.close();self.assertTrue(pool.closed)

    def test_focus_generation_cannot_deliver_old_result(self):
        pool=ManualPool();runner=AsyncPolicy(SimpleNamespace(propose=lambda r:None),PolicyConfig(mode='active'),pool)
        runner.submit(observed(),1,'scene',{});req=runner.request;runner.invalidate(1.1,'focus_lost')
        pool.calls[0][0].set_result(intent(ident=req.request_id))
        self.assertIsNone(runner.poll(1.2))

    def test_custom_provider_cannot_extend_request_validity(self):
        pool=ManualPool();runner=AsyncPolicy(SimpleNamespace(propose=lambda r:None),PolicyConfig(mode='active'),pool)
        runner.submit(observed(),1,'scene',{});req=runner.request
        pool.calls[0][0].set_result(replace(intent(ident=req.request_id),valid_until=100))
        self.assertIsNone(runner.poll(1.2))
        self.assertEqual(runner.events[-1]['event'],'policy_response_rejected')
        runner.close()

    def test_exception_details_do_not_leak_and_retry_backs_off(self):
        pool=ManualPool();runner=AsyncPolicy(SimpleNamespace(propose=lambda r:None),PolicyConfig(mode='active'),pool)
        runner.submit(observed(),1,'scene',{});pool.calls[0][0].set_exception(RuntimeError('private-key-value'))
        self.assertIsNone(runner.poll(1.1));self.assertNotIn('private-key-value',json.dumps(runner.events))
        self.assertFalse(runner.submit(observed(),1.2,'scene',{}))

    def test_responses_transport_payload_and_refusal_are_verified_offline(self):
        class Opener:
            def open(self,http,timeout):
                self.body=json.loads(http.data);self.timeout=timeout
                reply=dict(request_id='r',frame_id=1,map_epoch=0,action='wait',target='',world_x=None,
                           direction='',duration=.5,reason='wait')
                return io.StringIO(json.dumps(dict(status='completed',output=[dict(type='message',
                    content=[dict(type='output_text',text=json.dumps(reply))])])) )
        opener=Opener();policy=OpenAIActionPolicy(opener=opener)
        req=PolicyRequest('r','scene',1,0,1,10,{},(np.zeros((10,10,3),np.uint8),))
        with patch.dict('os.environ',{'OPENAI_API_KEY':'offline-test-value'}):
            self.assertEqual(policy.propose(req).action,'wait')
        self.assertFalse(opener.body['store']);self.assertTrue(opener.body['text']['format']['strict'])
        self.assertNotIn('offline-test-value',json.dumps(opener.body))
        self.assertEqual(opener.body['text']['format']['schema'],action_schema())


class FeedbackTests(unittest.TestCase):
    def test_floor_feedback_separates_idle_time_from_actual_attack_input(self):
        f=FeedbackTracker();o=observed(monsters=False);f.observe(o,1)
        f.hud(dict(exp=100,level=44),1,1);f.hud(dict(exp=100,level=44),1.2,1.2)
        o.captured_at=1.3;f.observe(o,1.3,previous_input={'held_keys':[]})
        o.captured_at=1.35;f.observe(o,1.35,previous_input={'held_keys':['shift']})
        state=f.status(1.35)['standing_observation']
        self.assertAlmostEqual(state['attack_input_seconds'],.05)
        self.assertAlmostEqual(state['seconds_here'],.35)
        self.assertEqual(state['net_exp_here'],0)
        f.hud(dict(exp=None,level=44),1.4,1.4)
        self.assertIsNone(f.status(1.4)['standing_observation']['net_exp_here'])
        f.invalidate();self.assertEqual(f.status(2)['standing_observation']['attack_input_seconds'],0)

    def test_firing_geometry_clips_range_height_and_task_masks(self):
        b=base();b.firing_platforms={'a'};b.combat_platforms={'b','below'}
        o=observed(monsters=False)
        o.player=Actor(Box(45,52,75,100),.99)
        o.platforms=[Platform('a',0,120,100),Platform('b',200,800,160),Platform('below',200,800,300)]
        e=ActionExecutor(b);snapshot=StateHistory().snapshot(o,1,(50,0),b,e,{},True)
        geometry=snapshot['firing_geometry'];self.assertEqual(len(geometry),1)
        spans=geometry[0]['possible_combat_spans'];self.assertEqual(len(spans),1)
        self.assertEqual(spans[0]['combat_platform'],'b');self.assertEqual(spans[0]['direction'],'right')
        margin=e.clearance(o)
        self.assertEqual(spans[0]['possible_target_world_x'],[150.,120-margin+b.motion.attack_max-50])
        self.assertEqual(geometry[0]['allowed_stand_world_x'],[margin-50,120-margin-50])

    def test_reward_window_uses_verified_readings_and_resets_on_gaps(self):
        f=FeedbackTracker()
        for t,exp in ((1,100),(1.2,100),(2,100),(2.2,100),(3,160),(3.2,160)):
            f.hud(dict(exp=exp,level=44),t,t)
        window=f.status(3.2)['experience_window']
        self.assertEqual(window['net_exp'],60)
        self.assertAlmostEqual(window['observed_seconds'],2)
        f.hud(dict(exp=None,level=44),3.4,3.4)
        self.assertIsNone(f.status(3.4)['experience_window'])
        for t in (8,8.2):f.hud(dict(exp=200,level=44),t,t)
        self.assertIsNone(f.status(8.2)['experience_window'])

    def test_snapshot_serializes_numpy_detection_comparisons(self):
        b=base();e=ActionExecutor(b);o=observed()
        o.monsters=[Actor(Box(np.float64(720),np.float64(172),np.float64(760),np.float64(220)),.99)]
        snapshot=StateHistory().snapshot(o,1,(0,0),b,e,{},True)
        self.assertIsInstance(snapshot['candidate_destinations'][0]['visible_nearby_monsters'],int)
        json.dumps(snapshot)

    def test_policy_sees_confirmation_and_expiry_of_experience_samples(self):
        f=FeedbackTracker()
        f.hud(dict(exp=100,level=44),1,1)
        self.assertFalse(f.status(1)['experience']['confirmed'])
        f.hud(dict(exp=100,level=44),1.2,1.2)
        self.assertTrue(f.status(1.2)['experience']['confirmed'])
        f.hud(dict(exp=150,level=44),1.4,1.4)
        self.assertFalse(f.status(1.4)['experience']['confirmed'])
        self.assertEqual(f.status(1.4)['confirmed_experience']['exp'],100)
        self.assertIsNone(f.status(4)['confirmed_experience'])
        f.invalidate();self.assertIsNone(f.status(4)['experience']['exp'])

    def test_confirmed_experience_missing_level_gap_and_loss(self):
        f=FeedbackTracker()
        def read(exp,level=44,t=1):return f.hud(dict(exp=exp,level=level),t,t)
        self.assertFalse(read(100));self.assertEqual(read(100,t=1.2)[0]['event'],'experience_baseline_confirmed')
        self.assertFalse(read(150,t=1.4));events=read(150,t=1.6)
        self.assertEqual(events[0]['net_exp'],50);self.assertEqual(events[0]['attack_attribution'],'unknown')
        self.assertEqual(read(None,t=1.8)[0]['event'],'experience_unknown')
        read(170,t=2);self.assertEqual(read(170,t=2.2)[0]['event'],'experience_baseline_confirmed')
        read(120,t=2.4);self.assertEqual(read(120,t=2.6)[0]['reason'],'net_loss_review_required')
        read(10,45,t=2.8);self.assertEqual(read(10,45,t=3)[0]['event'],'experience_baseline_confirmed')
        read(30,45,t=8);self.assertEqual(read(30,45,t=8.2)[0]['event'],'experience_baseline_confirmed')

    def test_target_loss_and_stuck_events_are_not_success_rewards(self):
        f=FeedbackTracker();o=observed();f.observe(o,1)
        o.monsters=[];o.captured_at=1.2;events=f.observe(o,1.2)
        self.assertTrue(all(e['kill'] is None for e in events if e['event']=='target_lost'))
        receipt=dict(held_keys=['right']);f.observe(o,1.2,previous_input=receipt)
        events=f.observe(o,2.4,previous_input=receipt)
        self.assertTrue(any(e['event']=='movement_no_observed_progress' for e in events))

    def test_history_is_bounded_world_relative_and_clears_on_epoch(self):
        h=StateHistory();o=observed()
        for i in range(200):
            o.captured_at=i*.1;h.observe(o,i*.1,(40,0),dict(held_keys=['right']))
        self.assertLessEqual(len(h.rows),22);self.assertEqual(h.rows[-1]['player_world'],[260,100])
        o.map_epoch=1;h.observe(o,21,(40,0),None);self.assertEqual(len(h.rows),1)

    def test_state_serialization_round_trips_entities_and_quantization(self):
        o=observed();self.assertEqual(observation_data(observation_from_data(observation_data(o))),observation_data(o))


class TraceTests(unittest.TestCase):
    def test_runtime_uses_pipeline_with_fake_game_and_fake_model(self):
        from autofarm.realtime.runtime import run
        from autofarm.realtime.perception import Frame
        from contextlib import nullcontext
        class WaitPolicy:
            def propose(self,r):
                return ActionIntent.from_reply(dict(request_id=r.request_id,frame_id=r.frame_id,map_epoch=r.map_epoch,
                    action='wait',target='',world_x=None,direction='',duration=.5,reason='offline fixture'),r)
        with tempfile.TemporaryDirectory() as root:
            folder=Path(root);(folder/'scene.json').write_text('{}')
            scene=SimpleNamespace(confidence=.99,request_id='scene',preferred=[],hud_boxes=[],
                                  width=700,height=120,map_name='fixture',play_area=Box(0,0,700,120))
            image=np.zeros((120,700,3),np.uint8)
            class Vision:
                def __init__(self,*a,**kw):
                    self.scene=scene;self.offset=(0,0);self.seed=image
                    self.identity_source='offline_fixture'
                    self.minimap=SimpleNamespace(status=lambda:{},to_data=lambda:{})
                def observe(self,packet,epoch):return observed(packet.started,packet.id)
                def annotate(self,*args):return image
                def close(self):pass
            class Capture:
                def __init__(self,*a):self.id=0;self.backend='offline_fixture';self.error=None
                def __enter__(self):return self
                def __exit__(self,*a):pass
                def next(self,*a):
                    time.sleep(.025);self.id+=1
                    if self.id==8:(folder/'STOP').write_text('offline fixture complete')
                    stamp=time.perf_counter()
                    return Frame(self.id,stamp,stamp,image,True,(0,0,700,120))
            api=SimpleNamespace(get_foreground=lambda:1)
            class Adapter:
                def __init__(self):self.sent=[]
                def send_key(self,key,up=False):self.sent.append((key,up))
                def is_target_foreground(self,hwnd):return True
                def emergency_pressed(self):return False
            adapter=Adapter()
            with patch('autofarm.realtime.runtime.load_scene',return_value=(scene,image)),\
                 patch('autofarm.realtime.runtime.GroundedVision',Vision),\
                 patch('autofarm.realtime.runtime.LatestCapture',Capture),\
                 patch('autofarm.realtime.runtime.InputSessionLock',return_value=nullcontext()):
                run(api,1,folder,seconds=1,live=True,adapter=adapter,policy_mode='active',action_policy=WaitPolicy())
            report=json.loads((folder/'report.json').read_text())
            self.assertGreaterEqual(report['action_policy']['counts'].get('accepted',0),1)
            self.assertFalse(adapter.sent);self.assertTrue(report['keys_released'])
            replayed=replay_trace(folder)
            self.assertTrue(replayed['identical'],replayed['first_mismatches'])

    def test_controller_kind_is_preserved_without_adding_navigation_calibration(self):
        controller=Controller(MotionProfile(180,100,150,True),True)
        self.assertIs(type(restore_controller(controller_configuration(controller,True))),Controller)

    def test_active_actions_receipts_focus_and_feedback_replay_identically(self):
        with tempfile.TemporaryDirectory() as root:
            b=base();b.applied_facing='right';b.applied_face_at=0
            # Initial direction is derived from a recorded accepted face action, not hidden state.
            b.applied_facing=None
            p=DecisionPipeline(b,PolicyConfig(mode='active'),folder=root,replay=True)
            o=observed();d=p.step(o,1,scene_id='scene',delivery=intent('face',direction='right'))
            p.commit(d,True,1,o,{'right'})
            o.captured_at=1.1;o.frame_id=2;d=p.step(o,1.1,scene_id='scene');p.commit(d,True,1.1,o)
            o.captured_at=1.2;o.frame_id=3
            d=p.step(o,1.2,scene_id='scene',delivery=intent('attack','7',t=1.2,frame=3,ident='shot'),
                     readings=dict(reading=dict(exp=100,level=44),sampled_at=1.2))
            p.commit(d,True,1.2,o,{'shift'})
            o.captured_at=1.4;o.frame_id=4
            d=p.step(o,1.4,scene_id='scene',readings=dict(reading=dict(exp=100,level=44),sampled_at=1.4))
            p.commit(d,True,1.4,o,{'shift'})
            p.reset(1.5,'focus_lost')
            d=p.step(None,1.5,scene_id='scene',override=Decision(reason='focus_lost'),override_source='focus_guard')
            p.commit(d,False,1.5)
            p.close(1.5)
            r=replay_trace(root);self.assertTrue(r['identical'],r['first_mismatches'])
            self.assertTrue(r['terminal_recorded']);self.assertTrue(r['terminal_matches'])
            self.assertEqual(r['cycles'],5);self.assertFalse(r['automatic_inputs'])

    def test_rule_trace_replays_acknowledged_attack_state(self):
        with tempfile.TemporaryDirectory() as root:
            b=Controller(MotionProfile(180,100,150,True),True);b.direct_attacks=True
            p=DecisionPipeline(b,navigate=True,folder=root,replay=True)
            for k in range(80):
                t=1+k/30;o=observed(t,k+1)
                d=p.step(o,t,scene_id='scene');p.commit(d,False,t,o,simulated=True)
            p.close(4)
            r=replay_trace(root);self.assertTrue(r['identical'],r['first_mismatches'])

    def test_missing_receipt_is_not_acknowledged_or_started(self):
        p=DecisionPipeline(base(),PolicyConfig(mode='active'),replay=True);o=observed()
        d=p.step(o,1,scene_id='scene',delivery=intent('move',x=450));p.commit(d,False,1,o)
        self.assertIsNone(p.executor.applied_at);self.assertFalse(p.last_trace['acknowledged'])

    def test_frame_guard_cancels_intent_and_replays_proposal_and_receipt(self):
        with tempfile.TemporaryDirectory() as root:
            p=DecisionPipeline(base(),PolicyConfig(mode='active'),folder=root,replay=True)
            o=observed();d=p.step(o,1,scene_id='scene',delivery=intent('face',direction='right'))
            self.assertEqual(d.keys,{'right'})
            p.commit(Decision(reason='stale_frame'),False,1,o)
            self.assertIsNone(p.executor.current);self.assertIsNone(p.base.applied_facing)
            self.assertEqual(p.last_trace['source'],'frame_guard')
            self.assertEqual(p.last_trace['proposed_decision']['keys'],['right'])
            o.captured_at=1.1;o.frame_id=2
            d=p.step(o,1.1,scene_id='scene',delivery=intent(t=1.1,frame=2,ident='next'))
            p.commit(d,False,1.1,o,simulated=True);p.close(1.2)
            r=replay_trace(root);self.assertTrue(r['identical'],r['first_mismatches'])

    def test_external_parking_receipt_does_not_change_rule_facing_in_replay(self):
        with tempfile.TemporaryDirectory() as root:
            p=DecisionPipeline(base(),PolicyConfig(mode='active'),folder=root,replay=True)
            parking=base();o=observed()
            d=p.step(o,1,scene_id='scene',override=Decision(frozenset({'left'}),'parking_fixture'),
                     override_source='parking')
            p.commit(d,True,1,o,{'left'},owner=parking)
            self.assertEqual(parking.applied_facing,'left');self.assertIsNone(p.base.applied_facing)
            o.captured_at=1.1;o.frame_id=2
            d=p.step(o,1.1,scene_id='scene',delivery=intent('attack','8',t=1.1,frame=2,ident='next'))
            self.assertEqual(d.keys,{'left'})
            p.commit(d,True,1.1,o,{'left'});p.close(1.2)
            r=replay_trace(root);self.assertTrue(r['identical'],r['first_mismatches'])

    def test_invalid_feedback_configuration_preserves_prior_trace(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'policy_trace.jsonl';path.write_text('prior trace',encoding='utf-8')
            (Path(root)/'feedback_config.json').write_text('{"version":99}',encoding='utf-8')
            with self.assertRaises(ValueError):DecisionPipeline(base(),folder=root)
            self.assertEqual(path.read_text(encoding='utf-8'),'prior trace')


if __name__=='__main__':unittest.main()
