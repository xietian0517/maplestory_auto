import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from autofarm.realtime.model import Actor,Box,Decision,MotionProfile,Observation,Platform,Scene,Rope
from autofarm.realtime.control import Controller,LeasedKeys,graph,route
from autofarm.realtime.perception import Camera,Frame,GroundedVision
from autofarm.realtime.semantic import atomic_json,load_request,make_request
from autofarm.realtime.runtime import Metrics


def scene_data():
    return dict(request_id='test',map_name='Unseen map',width=640,height=400,
                play_area=[0,0,640,360],player_name_box=[290,304,342,318],player_foot_offset=-12,
                monster_boxes=[[400,265,432,300]],exclude_boxes=[],
                platforms=[dict(id='p1',left=0,right=640,y=300)],ropes=[],
                preferred_platforms=['p1'],confidence=.9,reasoning='test')


def observation(x=300,monster_x=480,time=1):
    return Observation(1,time,Actor(Box(x-15,252,x+15,300),.99),
                       [Actor(Box(monster_x-20,260,monster_x+20,300),.95)] if monster_x else [],
                       [Platform('p1',0,640,300)],map_epoch=1)


class SceneTests(unittest.TestCase):
    def test_reject_cross_request(self):
        with self.assertRaises(ValueError): Scene.parse(scene_data(),'another',640,400)
    def test_reject_resize(self):
        with self.assertRaises(ValueError): Scene.parse(scene_data(),'test',1280,800)
    def test_reject_non_finite(self):
        data=scene_data(); data['player_foot_offset']=float('nan')
        with self.assertRaises(ValueError): Scene.parse(data,'test',640,400)
    def test_duplicate_platform(self):
        data=scene_data(); data['platforms']*=2
        with self.assertRaises(ValueError): Scene.parse(data,'test',640,400)
    def test_preference_must_exist(self):
        data=scene_data(); data['preferred_platforms']=['invented']
        with self.assertRaises(ValueError): Scene.parse(data,'test',640,400)
    def test_seed_integrity(self):
        with tempfile.TemporaryDirectory() as d:
            request=make_request(np.zeros((100,120,3),np.uint8),d)
            (Path(d)/request['image']).write_bytes(b'changed')
            with self.assertRaises(ValueError): load_request(d)


class ControlTests(unittest.TestCase):
    def test_turn_then_attack_then_target_disappears(self):
        c=Controller(); self.assertEqual(c.decide(observation(),1.01).reason,'turn')
        self.assertEqual(c.decide(observation(time=1.05),1.06).keys,frozenset({'shift'}))
        self.assertFalse(c.decide(observation(monster_x=None,time=1.1),1.11).keys)
    def test_stale_frame_cancels(self):
        c=Controller(); self.assertFalse(c.decide(observation(),1.101).keys)
    def test_wrong_height_not_attacked(self):
        o=observation(); o.monsters=[Actor(Box(460,100,500,140),.95)]
        self.assertFalse(Controller().decide(o,1.01).keys)
    def test_near_enemy_faces_target_then_jumps_before_attacking(self):
        c=Controller()
        self.assertEqual(c.decide(observation(monster_x=325),1.01).reason,'jump_attack_turn')
        self.assertEqual(c.decide(observation(monster_x=325,time=1.07),1.07).keys,{'alt'})
    def test_platform_edge_recovers(self):
        self.assertEqual(Controller().decide(observation(x=12),1.01).keys,frozenset({'right'}))
    def test_floor_unknown_waits_without_a_target(self):
        o=observation(monster_x=None); o.platforms=[]
        self.assertFalse(Controller().decide(o,1.01).keys)
    def test_uncalibrated_no_jump(self):
        p=[Platform('a',0,200,200),Platform('b',220,400,140)]
        self.assertEqual(graph(p,[],MotionProfile())['a'],[])
    def test_route_respects_failed_edge(self):
        edges={'a':[('b','jump')],'b':[('c','walk')],'c':[]}
        self.assertEqual(len(route(edges,'a','c')),2)
        self.assertEqual(route(edges,'a','c',{('a','b')}),[])
    def test_map_epoch_cancels_motion(self):
        c=Controller(); c.decide(observation(),1.01)
        c.transition=('p1','p2','jump'); o=observation(time=2); o.map_epoch=2
        c.decide(o,2.01); self.assertIsNone(c.transition)


class Adapter:
    def __init__(self): self.events=[]; self.focus=True; self.emergency=False
    def is_target_foreground(self,hwnd): return self.focus
    def emergency_pressed(self): return self.emergency
    def send_key(self,key,up): self.events.append((key,up))


class NavigationTests(unittest.TestCase):
    def rope_observation(self,x=300,y=500,t=1):
        return Observation(1,t,Actor(Box(x-15,y-48,x+15,y),.99),[],
            [Platform('lower',0,640,500),Platform('upper',200,500,300)],
            [Rope(300,300,450)],map_epoch=1)

    def test_jump_up_catches_rope_and_climbs(self):
        from autofarm.realtime.climbing import RopeClimber
        c=RopeClimber()
        self.assertEqual(c.decide(self.rope_observation(),1).keys,{'alt','up'})
        self.assertEqual(c.decide(self.rope_observation(y=430,t=1.5),1.5).reason,'rope_catch')
        self.assertFalse(c.grab_confirmed)
        c.decide(self.rope_observation(y=390,t=1.7),1.7)
        self.assertEqual(c.decide(self.rope_observation(y=380,t=1.84),1.84).reason,'rope_ascend')
        self.assertTrue(c.grab_confirmed)
        self.assertEqual(c.decide(self.rope_observation(y=300,t=2),2).reason,'rope_confirm_landing')
        self.assertEqual(c.decide(self.rope_observation(y=300,t=2.06),2.06).reason,'climb_complete')

    def test_knockback_on_rope_landing_is_success(self):
        from autofarm.realtime.climbing import RopeClimber
        c=RopeClimber(); c.decide(self.rope_observation(),1)
        c.decide(self.rope_observation(x=390,y=300,t=2),2)
        self.assertEqual(c.decide(self.rope_observation(x=395,y=300,t=2.06),2.06).reason,'climb_complete')

    def test_stale_rope_frame_never_holds_up(self):
        from autofarm.realtime.climbing import RopeClimber
        c=RopeClimber(); c.decide(self.rope_observation(),1)
        self.assertFalse(c.decide(self.rope_observation(),1.1).keys)

    def test_observed_drop_does_not_need_jump_estimate(self):
        p=[Platform('a',220,350,300),Platform('b',0,640,370)]
        self.assertIn(('b','drop'),graph(p,[],MotionProfile())['a'])

    def test_calibration_uses_relative_platform_position(self):
        from autofarm.realtime.calibration import Calibrator
        c=Calibrator()
        def feed(t,x=300,y=300,offset=0):
            o=observation(x=x+offset,monster_x=None,time=t)
            o.player=Actor(Box(x+offset-15,y-48,x+offset+15,y),.99)
            o.platforms=[Platform('p1',offset,640+offset,300)]
            return c.decide(o,t)
        feed(1); feed(1.26,x=350); feed(1.39,x=350,offset=30)
        feed(1.43); feed(1.60); feed(1.82,y=220); feed(2.02,y=240); feed(2.17)
        self.assertIsNotNone(c.result)
        self.assertAlmostEqual(c.result.speed,50/.26)
        self.assertEqual(c.result.jump_height,80)

    def test_failed_calibration_moves_on_to_another_floor(self):
        from autofarm.realtime.calibration import NavigationController
        c=NavigationController(); c.decide(observation(monster_x=None),1)
        c.calibrator.failed=True; c.calibrator.floor_id='p1'
        c.decide(observation(monster_x=None,time=1.05),1.05)
        c.decide(observation(monster_x=None,time=1.1),1.1)
        self.assertIsNone(c.calibrator)
        o=observation(monster_x=None,time=1.2); o.platforms=[Platform('p2',0,640,300)]
        c.decide(o,1.2); self.assertIsNotNone(c.calibrator)

    def test_devhost_rejects_external_path_and_expired_job(self):
        from game_ai_devhost import job_args
        job=dict(issued_at=100,command='play',folder='captures/test',seconds=10)
        self.assertIn('--seconds',job_args(job,101)[0])
        with self.assertRaises(ValueError): job_args(job,140)
        job['folder']='../outside'
        with self.assertRaises(ValueError): job_args(job,101)


class LeaseTests(unittest.TestCase):
    def setUp(self):
        self.t=1; self.adapter=Adapter(); self.keys=LeasedKeys(self.adapter,1,lambda:self.t)
    def test_jump_pulse_releases_at_40ms_without_waiting_for_vision(self):
        self.keys.apply({'alt'},1,0,pulses={'alt':(.04,123)})
        self.t=1.041;self.keys.check()
        self.assertFalse(self.keys.held)
        self.assertEqual(self.keys.pop_releases(),[('alt',1.041,123)])
        self.t=1.045;self.keys.apply({'alt'},1.04,0)
        self.assertFalse(self.keys.held)  # A delayed frame cannot re-press this jump.
        self.keys.apply(set(),1.04,0)
        self.t=1.1;self.keys.apply({'alt'},1.1,0,pulses={'alt':(.04,456)})
        self.assertEqual(self.keys.held,{'alt'})

    def test_jump_pulse_does_not_extend_focus_or_stale_frame_lease(self):
        self.keys.apply({'alt'},.94,0,pulses={'alt':(.04,123)})
        self.t=1.036;self.keys.check()
        self.assertFalse(self.keys.held)
        self.assertEqual(self.keys.pop_releases(),[('alt',1.036,123)])
        self.keys.apply(set(),1.036,0)
        self.keys.apply({'alt'},1.036,0,pulses={'alt':(.04,456)})
        self.adapter.focus=False;self.keys.check()
        self.assertFalse(self.keys.held)
        self.assertEqual(self.keys.pop_releases(),[('alt',1.036,456)])

    def test_direction_is_pressed_before_jump_in_a_chord(self):
        self.keys.apply({'alt','down'},1,0)
        self.assertEqual(self.adapter.events,[('down',False),('alt',False)])
    def test_expired_lease_releases_without_new_frames(self):
        self.keys.apply({'shift'},1,0); self.t=1.07; self.keys.check()
        self.assertEqual(self.adapter.events,[('shift',False),('shift',True)])
    def test_focus_change_invalidates_inflight_frame(self):
        self.keys.apply({'left'},1,0); self.adapter.focus=False; self.keys.check()
        self.adapter.focus=True; self.t=1.01
        self.assertFalse(self.keys.apply({'shift'},1,0)); self.assertFalse(self.keys.held)
    def test_opposite_direction_rejected(self):
        with self.assertRaises(ValueError): self.keys.apply({'left','right'},1,0)
    def test_f11_latches_stop(self):
        self.keys.apply({'shift'},1,0); self.adapter.emergency=True; self.keys.check()
        self.adapter.emergency=False
        self.assertFalse(self.keys.apply({'shift'},1,0)); self.assertTrue(self.keys.stopped.is_set())
    def test_stale_and_future_frame_rejected(self):
        self.assertFalse(self.keys.apply({'shift'},.9,0))
        self.assertFalse(self.keys.apply({'shift'},2,0))
    def test_same_state_does_not_repeat_keydown(self):
        self.keys.apply({'shift'},1,0); self.t=1.02; self.keys.apply({'shift'},1.02,0)
        self.assertEqual(self.adapter.events,[('shift',False)])
    def test_state_change_releases_previous(self):
        self.keys.apply({'left'},1,0); self.keys.apply({'shift'},1,0)
        self.assertEqual(self.adapter.events,[('left',False),('left',True),('shift',False)])


class VisionTests(unittest.TestCase):
    def test_explicit_pet_appearance_is_excluded(self):
        rng=np.random.default_rng(7); image=rng.integers(0,256,(400,640,3),dtype=np.uint8)
        data=scene_data(); data['exclude_boxes']=data['monster_boxes'].copy()
        v=GroundedVision(Scene.parse(data,'test',640,400),image)
        o=v.observe(Frame(1,1,1,image,True,()))
        self.assertEqual(o.monsters,[])
    def test_seed_grounding_and_missing_name(self):
        rng=np.random.default_rng(7)
        image=rng.integers(0,256,(400,640,3),dtype=np.uint8)
        scene=Scene.parse(scene_data(),'test',640,400)
        vision=GroundedVision(scene,image)
        o=vision.observe(Frame(1,1,1.001,image,True,(0,0,640,400)))
        self.assertIsNotNone(o.player); self.assertTrue(o.monsters)
        changed=image.copy(); changed[230:320,285:347]=0
        o=vision.observe(Frame(2,1.03,1.031,changed,True,(0,0,640,400)))
        self.assertIsNone(o.player); self.assertEqual(o.reason,'player_not_found')
    def test_resize_invalidates_vision(self):
        rng=np.random.default_rng(2); image=rng.integers(0,256,(400,640,3),dtype=np.uint8)
        v=GroundedVision(Scene.parse(scene_data(),'test',640,400),image)
        self.assertEqual(v.observe(Frame(1,1,1,image[:300],True,())).reason,'window_resized')
    def test_metrics_are_json_serializable(self):
        metrics=Metrics(); f=Frame(1,1,1.01,None,True,())
        metrics.add(f,1.2,Decision(),'stale_frame',False,190)
        report=metrics.summary(); json.dumps(report); self.assertEqual(report['over_100ms'],1)


if __name__=='__main__': unittest.main()
