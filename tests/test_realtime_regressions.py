"""Real captured failure frames plus complete navigation state transitions."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import time

import cv2
import numpy as np

from autofarm.realtime.model import Actor,Box,MotionProfile,Observation,Platform,Rope,Scene
from autofarm.realtime.control import Controller,standing_platform,graph
from autofarm.realtime.climbing import RopeClimber
from autofarm.realtime.perception import Frame,GroundedVision
from autofarm.realtime.evidence import EvidenceRecorder
from autofarm.realtime.replay import replay
from autofarm.realtime.semantic import atomic_json,make_request


FIXTURES=Path(__file__).parent/'fixtures'/'realtime'


class CapturedRegressionTests(unittest.TestCase):
    def setUp(self):
        cv2.setNumThreads(2)
        self.seed=cv2.imread(str(FIXTURES/'seed.png'))
        data=json.loads((FIXTURES/'scene.json').read_text(encoding='utf-8'))
        self.scene=Scene.parse(data,data['request_id'],1366,768)
        self.vision=GroundedVision(self.scene,self.seed)

    def test_pet_label_occlusion_still_finds_visible_name(self):
        self.vision.observe(Frame(1,1,1,self.seed,True,()))
        lost=cv2.imread(str(FIXTURES/'pet_occlusion.png'))
        o=self.vision.observe(Frame(2,1.04,1.04,lost,True,()))
        self.assertIsNotNone(o.player)
        self.assertAlmostEqual(o.player.box.cx,683.5,delta=2)
        self.assertAlmostEqual(o.player.box.y2,430,delta=2)

    def test_camera_anchor_corrects_accumulated_error(self):
        moved=cv2.imread(str(FIXTURES/'camera_scroll.png'))
        self.vision.camera.offset=np.array([-44.,7.])
        anchor=self.vision.camera.anchor(moved)
        self.assertIsNotNone(anchor)
        self.assertAlmostEqual(anchor[0],-71,delta=3)
        self.assertAlmostEqual(anchor[1],16,delta=3)

    def test_full_occlusion_does_not_reuse_old_position(self):
        self.vision.observe(Frame(1,1,1,self.seed,True,()))
        image=self.seed.copy(); image[350:470,620:715]=0
        o=self.vision.observe(Frame(2,1.04,1.04,image,True,()))
        self.assertIsNone(o.player)

    def test_hidden_name_keeps_body_identity(self):
        image=self.seed.copy(); b=self.scene.name_box
        image[round(b.y1):round(b.y2),round(b.x1):round(b.x2)]=0
        o=self.vision.observe(Frame(1,1,1,image,True,()))
        self.assertIsNotNone(o.player)
        self.assertEqual(self.vision.identity_source,'appearance')
        self.assertAlmostEqual(o.player.box.cx,b.cx,delta=2)

    def test_global_appearance_reacquires_outside_previous_search(self):
        tracker=self.vision.appearance; pose=tracker.poses[0]
        image=np.zeros_like(self.seed); image[200:265,950:1002]=pose
        self.assertIsNone(tracker.locate(image,665,430))
        hit=tracker.rank(image,(665,430))
        self.assertIsNotNone(hit)
        self.assertAlmostEqual(hit[0],976,delta=2)
        self.assertAlmostEqual(hit[1],265,delta=2)

    def test_global_identity_does_not_choose_between_identical_sprites(self):
        tracker=self.vision.appearance; pose=tracker.poses[0]
        image=np.zeros_like(self.seed)
        image[200:265,450:502]=pose; image[200:265,750:802]=pose
        self.assertIsNone(tracker.rank(image,(665,430)))

    def test_ranked_candidate_needs_second_visual_observation(self):
        image=np.zeros_like(self.seed); image[200:265,950:1002]=self.vision.appearance.poses[0]
        first=self.vision._player(image,0,0)
        self.assertIsNotNone(first); self.assertTrue(self.vision.identity_pending)
        second=self.vision._player(image,0,0)
        self.assertIsNotNone(second); self.assertFalse(self.vision.identity_pending)
        self.assertIn(self.vision.identity_source,('appearance_reacquired','appearance','appearance_tracked'))

    def test_background_global_search_is_confirmed_on_a_fresh_frame(self):
        v=GroundedVision(self.scene,self.seed,async_reacquire=True)
        try:
            image=np.zeros_like(self.seed); image[200:265,950:1002]=v.appearance.poses[0]
            self.assertIsNone(v._player(image,0,0))
            v.reacquire_future.result(timeout=2)
            p=v._player(image,0,0)
            self.assertIsNotNone(p); self.assertAlmostEqual(p.box.cx,976,delta=2)
        finally: v.close()

    def test_repeated_platform_cannot_teleport_camera_anchor(self):
        camera=self.vision.camera
        with patch.object(camera,'anchor',return_value=np.array([350.,0.])):
            valid,offset=camera.update(self.seed)
        self.assertTrue(valid); self.assertAlmostEqual(offset[0],0,delta=1)

    def test_ui_exclusion_does_not_supply_world_anchors(self):
        for point in self.vision.camera.anchor_points:
            x,y=point.pt
            self.assertFalse(0<=x<63 and 0<=y<92)

    def test_platform_anchors_recover_real_parallax_drift(self):
        data=json.loads((FIXTURES/'parallax_scene.json').read_text(encoding='utf-8'))
        scene=Scene.parse(data,data['request_id'],1366,768)
        v=GroundedVision(scene,cv2.imread(str(FIXTURES/'parallax_seed.png')))
        v.camera.offset=np.array([191.3,-139.76])
        offset=v.camera.anchor(cv2.imread(str(FIXTURES/'parallax_drift.png')))
        self.assertIsNotNone(offset)
        self.assertAlmostEqual(offset[0],214,delta=4)
        self.assertAlmostEqual(offset[1],-184,delta=3)

    def test_changed_background_and_vine_covered_name_keep_sprite_identity(self):
        data=json.loads((FIXTURES/'parallax_scene.json').read_text(encoding='utf-8'))
        scene=Scene.parse(data,data['request_id'],1366,768)
        v=GroundedVision(scene,cv2.imread(str(FIXTURES/'parallax_seed.png')))
        image=cv2.imread(str(FIXTURES/'climb_name_covered.png'))
        v.player_last=(548,403-scene.foot_offset)
        player=v._player(image,0,0)
        self.assertIsNotNone(player)
        self.assertEqual(v.identity_source,'appearance_tracked')
        self.assertAlmostEqual(player.box.cx,548,delta=2)
        self.assertAlmostEqual(player.box.y2,403,delta=2)

    def test_old_partial_identity_expires(self):
        self.vision.observe(Frame(1,1,1,self.seed,True,()))
        lost=cv2.imread(str(FIXTURES/'pet_occlusion.png'))
        lost[360:405,650:720]=0  # No independent appearance evidence either.
        o=self.vision.observe(Frame(2,2,2,lost,True,()))
        self.assertIsNone(o.player)

    def test_recording_is_bounded_and_replays_without_input(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder); req=make_request(self.seed,p)
            data=self.scene.to_data(); data['request_id']=req['request_id']; atomic_json(p/'scene.json',data)
            scene=Scene.parse(data,req['request_id'],1366,768)
            with EvidenceRecorder(p,limit=2,hz=30) as recording:
                recording.bind_scene(scene,self.seed)
                for i in range(4):
                    recording.offer(Frame(i,1+i*.1,1+i*.1,self.seed,True,()),dict(frame_id=i,scene_id=scene.request_id))
            self.assertIsNone(recording.error)
            report=replay(p)
            self.assertEqual(report['frames'],2); self.assertEqual(report['visible_frames'],2)
            self.assertEqual(report['mode'],'OFFLINE_REPLAY')

    def test_runtime_final_status_and_async_recording(self):
        from autofarm.realtime.runtime import run
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder); req=make_request(self.seed,p)
            data=self.scene.to_data(); data['request_id']=req['request_id']; atomic_json(p/'scene.json',data)
            seed=self.seed
            class FakeCapture:
                backend='fixture'; error=None
                def __init__(self,*args): self.i=0
                def __enter__(self): return self
                def __exit__(self,*args): pass
                def next(self,*args):
                    self.i+=1
                    if self.i>3: (p/'STOP').write_text('done'); return None
                    t=time.perf_counter(); return Frame(self.i,t,t,seed,True,(0,0,1366,768))
            class FakeApi:
                def get_foreground(self): return 7
            with patch('autofarm.realtime.runtime.LatestCapture',FakeCapture):
                run(FakeApi(),7,p,seconds=1,record=True)
            report=json.loads((p/'report.json').read_text())
            status=json.loads((p/'status.json').read_text())
            self.assertEqual(status['phase'],'file_stop')
            self.assertTrue(report['keys_released']); self.assertEqual(report['active_input_frames'],0)
            self.assertGreater(report['recording_frames'],0); self.assertIsNone(report['recording_error'])

    def test_runtime_capture_failure_is_not_reported_complete(self):
        from autofarm.realtime.runtime import run
        class BrokenCapture:
            def __init__(self,*args): pass
            def __enter__(self): raise RuntimeError('capture failed')
            def __exit__(self,*args): pass
        with tempfile.TemporaryDirectory() as folder,patch('autofarm.realtime.runtime.LatestCapture',BrokenCapture):
            with self.assertRaises(RuntimeError): run(None,7,folder,seconds=1)
            report=json.loads((Path(folder)/'report.json').read_text())
            self.assertEqual(report['phase'],'error'); self.assertIn('capture failed',report['error'])


def nav_observation(t,x=200,y=300,monsters=(),vy=0):
    return Observation(round(t*100),t,Actor(Box(x-15,y-48,x+15,y),.99,vy=vy),list(monsters),
        [Platform('spawn',140,260,300),Platform('ground',0,640,400),Platform('combat',350,620,210)],
        [Rope(420,210,350)],map_epoch=1)


class NavigationRegressionTests(unittest.TestCase):
    def test_attack_direction_requires_accepted_turn_even_when_planner_thinks_turned(self):
        from autofarm.realtime.model import Decision
        c=Controller();c.facing='right';c.applied_facing='left'
        o=nav_observation(1,y=400,monsters=[Actor(Box(310,355,350,400),.99,track_id=7)])
        attack=Decision(frozenset({'shift'}),'jump_attack_fire_first','7')
        turn=c.orient_attack(o,attack,1)
        self.assertEqual(turn.keys,{'right'})
        o.captured_at=1.2
        self.assertEqual(c.orient_attack(o,attack,1.2).keys,{'right'})  # Unsent turn never completes.
        c.acknowledge(turn,1.2)
        o.captured_at=1.23
        self.assertNotIn('shift',c.orient_attack(o,attack,1.23).keys)
        o.captured_at=1.27
        self.assertEqual(c.orient_attack(o,attack,1.27),attack)
        c.acknowledge(Decision(frozenset({'left'}),'active_recovery_step'),1.28)
        o.captured_at=1.4
        self.assertEqual(c.orient_attack(o,attack,1.4).keys,{'right'})

    def test_crossing_target_releases_attack_and_changes_direction(self):
        from autofarm.realtime.model import Decision
        c=Controller();c.acknowledge(Decision(frozenset({'left'})),1)
        attack=Decision(frozenset({'shift'}),'jump_attack_hold','7')
        o=nav_observation(1.2,y=375,monsters=[Actor(Box(230,355,270,400),.99,track_id=7)])
        self.assertEqual(c.orient_attack(o,attack,1.2).keys,{'right'})
        o.monsters=[]
        self.assertFalse(c.orient_attack(o,attack,1.2).keys)

    def test_jump_target_follows_nearest_current_enemy_not_old_screen_position(self):
        from autofarm.realtime.combat import JumpAttack
        original=Actor(Box(20,355,60,400),.99,track_id=1)
        c=JumpAttack(Platform('ground',0,640,400),original,1,'left',400)
        # Camera scroll / target disappearance: a fresh nearby right monster wins.
        right=Actor(Box(230,355,270,400),.99,track_id=3)
        far_left=Actor(Box(0,355,30,400),.99,track_id=2)
        d=c.decide(nav_observation(1,y=400,monsters=[right,far_left]),1)
        self.assertEqual(d.keys,{'right'})
        self.assertEqual(c.target_id,3)

    def test_random_jump_early_timing_releases_alt_then_waits_30ms_without_visual_rise(self):
        from autofarm.realtime.combat import JumpAttack
        target=Actor(Box(310,355,350,400),.99)
        c=JumpAttack(Platform('ground',0,640,400),target,1,'right',400)
        def step(t): return c.decide(nav_observation(t,y=400,monsters=[target]),t)
        c.on_input_applied(step(1),1)
        self.assertEqual(step(1.02).keys,{'alt'})
        release=step(1.041)
        self.assertFalse(release.keys)
        self.assertNotIn('shift',step(1.075).keys)  # Release request not yet accepted.
        c.on_input_applied(release,1.041)
        self.assertNotIn('shift',step(1.06).keys)
        self.assertEqual(step(1.072).reason,'jump_attack_fire_first')
        self.assertNotIn('shift',step(1.4).keys)  # Never wait until a late falling shot.

    def test_late_frame_can_recover_an_aligned_shot_but_not_shoot_over_target(self):
        from autofarm.realtime.combat import JumpAttack
        target=Actor(Box(310,355,350,400),.99)
        c=JumpAttack(Platform('ground',0,640,400),target,1,'right',400)
        c.on_input_applied(c.decide(nav_observation(1,y=400,monsters=[target]),1),1)
        c.jump_released_at=1.04
        d=c.decide(nav_observation(1.22,y=375,vy=-100,monsters=[target]),1.22)
        self.assertIn('shift',d.keys);self.assertEqual(c.fire_mode,'early_recovery')
        self.assertNotIn('shift',c.decide(nav_observation(1.23,y=320,monsters=[target]),1.23).keys)
        self.assertNotIn('shift',c.decide(nav_observation(1.24,y=375),1.24).keys)

    def test_confirm_direction_before_jump_not_after_takeoff(self):
        from autofarm.realtime.combat import JumpAttack
        from autofarm.realtime.model import Decision
        target=Actor(Box(310,355,350,400),.99,track_id=7)
        c=Controller();c.applied_facing='left'
        c.jump_combat=JumpAttack(Platform('ground',0,640,400),target,1,'right',400)
        o=nav_observation(1,y=400,monsters=[target])
        takeoff=Decision(frozenset({'alt'}),'jump_attack_takeoff')
        turn=c.orient_attack(o,takeoff,1)
        self.assertEqual(turn.keys,{'right'})
        c.acknowledge(turn,1)
        o.captured_at=1.07
        self.assertEqual(c.orient_attack(o,takeoff,1.07),takeoff)

    def test_recent_monster_pose_recheck_uses_current_texture_and_colour(self):
        from autofarm.realtime.monster_tracking import recheck
        before=cv2.imread(str(FIXTURES/'monster_before_hit.png'))
        current=cv2.imread(str(FIXTURES/'monster_hit_pose.png'))
        box=Box(557,583,608,634);sample=before[583:634,557:608]
        hit=recheck(current,sample,box)
        self.assertIsNotNone(hit)
        self.assertAlmostEqual(hit.box.cx,box.cx,delta=2)
        empty=current.copy();empty[563:654,527:638]=current[563:654,800:911]
        self.assertIsNone(recheck(empty,sample,box))
        # Unchanged feet alone cannot confirm the monster if the upper body disappeared.
        partial=current.copy();partial[583:614,557:608]=current[583:614,800:851]
        self.assertIsNone(recheck(partial,sample,box))

    def test_jump_attack_height_controls_fire_delay_and_requires_applied_jump(self):
        from autofarm.realtime.combat import JumpAttack
        for feet,early in ((400,True),(340,False)):
            target=Actor(Box(310,feet-45,350,feet),.99)
            c=JumpAttack(Platform('ground',0,640,400),target,1,'right',400)
            d=c.decide(nav_observation(1,y=400,monsters=[target]),1)
            self.assertEqual(d.keys,{'alt'})
            self.assertIsNone(c.jumped_at)
            c.on_input_applied(d,1)
            d=c.decide(nav_observation(1.04,y=388,vy=-200,monsters=[target]),1.04)
            self.assertNotIn('shift',d.keys)
            c.on_input_applied(d,1.04)
            if early:
                release=c.decide(nav_observation(1.05,y=388,vy=-200,monsters=[target]),1.05)
                c.on_input_applied(release,1.05)
            d=c.decide(nav_observation(1.1,y=370,vy=-200,monsters=[target]),1.1)
            self.assertIn('shift',d.keys)
            self.assertNotIn('shift',c.decide(nav_observation(1.16,y=350,vy=-100),1.16).keys)

    def test_two_separate_acknowledged_shots_across_jumps(self):
        from autofarm.realtime.combat import JumpAttack
        target=Actor(Box(210,355,250,400),.99)
        c=JumpAttack(Platform('ground',0,640,400),target,1,'right',400)
        def step(t,y,vy=0):
            return c.decide(nav_observation(t,y=y,vy=vy,monsters=[target]),t)
        c.on_input_applied(step(1,400),1)
        self.assertNotIn('shift',step(1.04,388,-200).keys)
        c.on_input_applied(step(1.05,388,-200),1.05)
        first=step(1.1,375,-200)
        self.assertEqual(first.reason,'jump_attack_fire_first')
        self.assertEqual(c.shots_issued,0)  # Rejected input cannot count as a shot.
        c.on_input_applied(first,1.1);c.on_input_applied(first,1.11)
        self.assertEqual(c.shots_issued,1)
        self.assertNotIn('shift',step(1.35,320).keys)
        self.assertEqual(step(1.8,400).reason,'jump_attack_followup_jump')
        self.assertFalse(c.done)
        c.on_input_applied(step(1.9,400),1.9)
        c.on_input_applied(step(1.95,388,-200),1.95)
        second=step(2,375,-200)
        self.assertEqual(second.reason,'jump_attack_fire_second')
        c.on_input_applied(second,2)
        self.assertEqual(c.shots_issued,2)
        self.assertEqual(step(2.6,400).reason,'jump_attack_landed')
        self.assertTrue(c.done)

    def test_ranged_same_level_enemy_uses_default_live_jump_policy(self):
        c=Controller(MotionProfile(180,90,150,True),True);c.prefer_jump_attacks=True
        d=c.decide(nav_observation(1,y=400,monsters=[Actor(Box(350,355,390,400),.99)]),1)
        self.assertIsNotNone(c.jump_combat)
        self.assertNotIn('shift',d.keys)

    def test_rope_too_high_is_rejected_and_distant_rope_is_approached(self):
        o=nav_observation(1,x=200,y=400)
        o.ropes=[Rope(420,210,300)]
        c=RopeClimber('combat',jump_height=90,speed=180,jump_distance=100)
        self.assertEqual(c.decide(o,1).reason,'no_accessible_rope')
        o.ropes=[Rope(420,210,340)]
        c=RopeClimber('combat',jump_height=90,speed=180,jump_distance=100)
        self.assertEqual(c.decide(o,1).keys,{'right'})
        o.player=Actor(Box(365,352,395,400),.99)
        self.assertEqual(c.decide(o,1).reason,'rope_approach')  # 40px too far for this high rope end.

    def test_haste_refresh_uses_120_seconds_and_successful_submission(self):
        from autofarm.realtime.combat import HasteRefresh
        from autofarm.realtime.model import Decision
        h=HasteRefresh();c=Controller();d=Decision(reason='search')
        first=h.apply(nav_observation(1,y=400),d,1,c)
        self.assertEqual(first.keys,{'home'})
        self.assertEqual(h.next_at,0)
        h.on_input_applied(first,1)
        self.assertEqual(h.apply(nav_observation(120,y=400),d,120,c),d)
        self.assertEqual(h.apply(nav_observation(121,y=400),d,121,c).keys,{'home'})
        self.assertEqual(h.apply(nav_observation(122,y=375,vy=-200),d,122,c),d)

    def test_failed_jump_edge_is_not_retried_after_two_failures(self):
        c=Controller(MotionProfile(180,90,150,True),True);c.last_epoch=1
        c.fail_edge(('ground','combat'),1,'jump')
        c.fail_edge(('ground','combat'),3,'jump')
        c.decide(nav_observation(100,y=400),100)
        self.assertIn(('ground','combat'),c.failures)

    def test_brief_target_occlusion_preserves_second_shot_without_blind_fire(self):
        from autofarm.realtime.combat import JumpAttack
        target=Actor(Box(310,355,350,400),.99)
        c=JumpAttack(Platform('ground',0,640,400),target,1,'right',400)
        d=c.decide(nav_observation(1,y=400,monsters=[target]),1);c.on_input_applied(d,1)
        lost=c.decide(nav_observation(1.1,y=375,vy=-200),1.1)
        self.assertFalse(lost.keys);c.on_input_applied(lost,1.1)
        self.assertFalse(c.done)
        self.assertIn('shift',c.decide(nav_observation(1.15,y=370,vy=-200,monsters=[target]),1.15).keys)

    def test_enemy_interrupts_rope_approach_before_takeoff(self):
        c=Controller(MotionProfile(180,90,150,True),True);c.last_epoch=1;c.prefer_jump_attacks=True
        c.rope_climber=RopeClimber('combat');c.rope_climber.phase='approach'
        c.decide(nav_observation(1,y=400,monsters=[Actor(Box(310,355,350,400),.99)]),1)
        self.assertIsNone(c.rope_climber)
        self.assertIsNotNone(c.jump_combat)

    def test_jump_attack_does_not_interrupt_rope_catching(self):
        c=Controller(MotionProfile(180,90,150,True),True); c.last_epoch=1
        c.rope_climber=RopeClimber(target_id='combat')
        o=nav_observation(1,x=420,y=400,monsters=[Actor(Box(445,295,480,340),.99)])
        c.decide(o,1)
        self.assertIsNone(c.jump_combat)

    def test_no_target_patrol_is_interrupted_for_another_floor(self):
        from autofarm.realtime.recovery import ActiveRecovery
        from autofarm.realtime.model import Decision
        c=Controller(MotionProfile(180,90,150,True),True);c.last_epoch=1;c.preferred=['ground']
        r=ActiveRecovery();d=Decision(frozenset({'right'}),'patrol')
        r.apply(nav_observation(1,y=400),d,1,800,controller=c)
        o=nav_observation(7.1,y=400)
        self.assertEqual(r.apply(o,d,7.1,800,controller=c).reason,'engagement_timeout_replan')
        self.assertEqual(c.explore_origin,'ground')
        self.assertNotEqual(c.decide(o,7.1).reason,'patrol')

    def test_vertical_target_outside_firing_height_does_not_trigger_air_shots(self):
        c=Controller(MotionProfile(180,90,150,True),True)
        o=nav_observation(1,monsters=[Actor(Box(310,200,350,240),.99)])
        self.assertNotIn('shift',c.decide(o,1).keys)
        o.captured_at=1.1
        self.assertNotIn('shift',c.decide(o,1.1).keys)

    def test_empty_preferred_floor_does_not_force_unnecessary_floor_hopping(self):
        c=Controller(MotionProfile(180,90,150,True),True);c.last_epoch=1;c.no_enemy_since=0
        c.preferred=['ground','combat']
        d=c.decide(nav_observation(2,x=200,y=400),2)
        self.assertEqual(d.reason,'patrol')

    def test_failed_small_ledge_drop_walks_off_over_observed_lower_floor(self):
        c=Controller(MotionProfile(180,90,150,True),True);c.last_epoch=1;c.no_enemy_since=0
        c.preferred=['ground'];c.failure_counts[('spawn','ground')]=1
        d=c.decide(nav_observation(2),2)
        self.assertEqual(d.reason,'walk_off_drop');self.assertEqual(d.keys,{'left'})

    def test_full_map_monster_selects_lower_floor_before_empty_patrol(self):
        c=Controller(MotionProfile(180,90,150,True),True); c.last_epoch=1
        c.no_enemy_since=0; c.preferred=['ground']
        o=nav_observation(2,y=400)
        o.platforms.append(Platform('lower',0,640,580))
        o.navigation_targets=[Actor(Box(300,530,350,580),.95)]
        c.decide(o,2)
        self.assertEqual(c.pending_edge,('ground','lower','drop'))

    def test_dynamic_support_extends_clipped_floor_without_crossing_gap(self):
        from autofarm.realtime.terrain import LocalTerrain
        im=cv2.imread(str(FIXTURES/'frontier_platform.png'))
        terrain=LocalTerrain(im,[Platform('clipped',260,665,463)])
        found=terrain.support(im,210,463)
        self.assertIsNotNone(found); self.assertLess(found[0],210)
        updated=terrain.update(im,Actor(Box(195,415,225,463),.99),[Platform('clipped',260,665,463)],(0,0),1)
        self.assertLess(updated[0].left,210)
        broken=im.copy(); broken[450:505,220:300]=0
        edge=terrain.support(broken,210,463)
        self.assertTrue(edge is None or edge[1]<=220)
        self.assertIsNone(terrain.support(im,95,463))  # Name/guild overlays hide support.

    def test_wait_recovery_moves_briefly_and_yields_to_combat(self):
        from autofarm.realtime.recovery import ActiveRecovery
        from autofarm.realtime.model import Decision
        r=ActiveRecovery(); d=Decision(reason='airborne_or_floor_unknown')
        r.apply(nav_observation(1,x=200,y=550),d,1,800)
        action=r.apply(nav_observation(1.9,x=200,y=550),d,1.9,800)
        self.assertEqual(action.keys,{'right'})
        attack=Decision(frozenset({'shift'}),'attack')
        self.assertEqual(r.apply(nav_observation(1.95),attack,1.95,800),attack)
        r.reset(); self.assertFalse(r.apply(nav_observation(2),d,3,800).keys)

    def test_session_pose_bank_survives_scene_refresh(self):
        from types import SimpleNamespace
        from autofarm.realtime.runtime import restore_appearance
        def tracker(value):
            return SimpleNamespace(appearance=SimpleNamespace(poses=[np.full((65,52,3),value,np.uint8)]))
        with tempfile.TemporaryDirectory() as folder:
            np.save(Path(folder)/'identity_poses.npy',np.stack([tracker(1).appearance.poses[0],tracker(2).appearance.poses[0]]))
            first=tracker(3); restore_appearance(first,folder)
            self.assertEqual([int(p[0,0,0]) for p in first.appearance.poses],[1,2,3])
            updated=tracker(4); restore_appearance(updated,folder,first)
            self.assertEqual([int(p[0,0,0]) for p in updated.appearance.poses],[1,2,4])

    def test_identity_loss_never_invents_an_attack_target(self):
        c=Controller(MotionProfile(180,90,150,True),True)
        c.decide(nav_observation(1),1)
        o=nav_observation(1.1); o.player=None; o.reason='player_not_found'
        self.assertFalse(c.decide(o,1.1).keys)
        o.captured_at=2.6
        self.assertFalse(c.decide(o,2.6).keys)
        c.decide(nav_observation(3),3)
        o.captured_at=3.1; o.motion_valid=False
        self.assertFalse(c.decide(o,3.1).keys)

    def test_stationary_rope_pose_recovers_after_scene_refresh(self):
        c=Controller(MotionProfile(180,90,150,True),True)
        self.assertFalse(c.decide(nav_observation(1,x=420,y=310),1).keys)
        d=c.decide(nav_observation(1.2,x=420,y=310),1.2)
        self.assertEqual(d.reason,'rope_resume'); self.assertEqual(d.keys,{'up'})
        falling=Controller(MotionProfile(180,90,150,True),True)
        falling.decide(nav_observation(1,x=420,y=310,vy=200),1)
        self.assertFalse(falling.decide(nav_observation(1.2,x=420,y=340,vy=200),1.2).keys)

    def test_identity_interrupt_does_not_restart_navigation_search_delay(self):
        c=Controller(MotionProfile(180,90,150,True),True); c.last_epoch=1
        c.preferred=['ground']; c.no_enemy_since=0
        missing=nav_observation(2); missing.player=None; missing.reason='player_not_found'
        self.assertFalse(c.decide(missing,2).keys)
        missing.captured_at=2.5
        self.assertFalse(c.decide(missing,2.5).keys)
        self.assertEqual(c.no_enemy_since,0)
        self.assertEqual(c.decide(nav_observation(2.6),2.6).reason,'drop')

    def test_failed_exit_retries_instead_of_permanent_short_ledge_patrol(self):
        c=Controller(MotionProfile(180,90,150,True),True); c.last_epoch=1
        c.no_enemy_since=0
        c.fail_edge(('spawn','ground'),1)
        self.assertIn(('spawn','ground'),c.failures)
        self.assertEqual(c.decide(nav_observation(1.7),1.7).reason,'route_retry_pending')
        c.decide(nav_observation(2.1),2.1)
        self.assertNotIn(('spawn','ground'),c.failures)
        self.assertEqual(c.transition,('spawn','ground','drop'))

    def test_target_across_gap_interrupts_pending_route(self):
        c=Controller(navigate=True); c.last_epoch=1
        c.pending_edge=('spawn','ground','drop'); c.pending_since=0
        o=nav_observation(1,monsters=[Actor(Box(310,250,350,295),.99)])
        self.assertEqual(c.decide(o,1).reason,'turn')
        o.captured_at=1.06
        self.assertEqual(c.decide(o,1.06).keys,{'shift'})

    def test_missing_floor_annotation_does_not_block_visible_target_attack(self):
        c=Controller(navigate=True)
        o=nav_observation(1,y=550,monsters=[Actor(Box(290,505,330,550),.99)])
        self.assertEqual(c.decide(o,1).reason,'turn_without_floor_map')
        o.captured_at=1.06
        self.assertEqual(c.decide(o,1.06).keys,{'shift'})

    def test_narrow_platform_patrol_does_not_flip_sides_forever(self):
        c=Controller(navigate=True); c.last_epoch=1; c.no_enemy_since=0
        o=nav_observation(2,x=200); o.platforms=o.platforms[:1]
        d=c.decide(o,2)
        self.assertEqual(d.reason,'patrol'); self.assertTrue(d.keys)

    def test_calibration_can_land_on_an_overlapping_higher_platform(self):
        from autofarm.realtime.calibration import Calibrator
        c=Calibrator(); c.phase='jump'; c.jump_at=1; c.peak_at=1.3
        c.peak=90; c.speed=150; c.floor_id='ground'; c.last_epoch=1
        self.assertEqual(c.decide(nav_observation(1.5,x=200,y=300),1.5).reason,'calibrate_jump')
        c.decide(nav_observation(1.6,x=200,y=300),1.6)
        self.assertIsNotNone(c.result); self.assertTrue(c.result.calibrated)

    def test_observed_cruising_refines_underestimated_short_walk(self):
        from autofarm.realtime.calibration import NavigationController
        from autofarm.realtime.model import Decision
        c=NavigationController(MotionProfile(80,100,55,True))
        with patch.object(c.base,'decide',return_value=Decision(frozenset({'right'}),'patrol')):
            for i in range(12):
                t=1+i*.04
                c.decide(nav_observation(t,x=100+i*6,y=400),t)
        self.assertAlmostEqual(c.base.motion.speed,150,delta=1)
        self.assertGreater(c.base.motion.jump_distance,100)
        c.reset()
        self.assertFalse(c.travel_samples); self.assertIsNone(c.last_action)

    def test_short_slow_start_does_not_reduce_jump_reach(self):
        from autofarm.realtime.calibration import NavigationController
        from autofarm.realtime.model import Decision
        c=NavigationController(MotionProfile(180,90,120,True))
        with patch.object(c.base,'decide',return_value=Decision(frozenset({'right'}),'patrol')):
            for i in range(12):
                t=1+i*.04
                c.decide(nav_observation(t,x=100+i*3,y=400),t)
        self.assertEqual(c.base.motion.speed,180)
        self.assertEqual(c.base.motion.jump_distance,120)

    def test_drop_rope_landing_then_attack(self):
        c=Controller(MotionProfile(220,80,160,True),True); c.preferred=['combat']
        self.assertEqual(c.decide(nav_observation(1),1).reason,'search')
        d=c.decide(nav_observation(2.6),2.6); self.assertEqual(d.reason,'drop')
        self.assertEqual(d.keys,{'down'})
        self.assertEqual(c.decide(nav_observation(2.75),2.75).keys,{'down','alt'})
        c.decide(nav_observation(2.9,y=350,vy=250),2.9)
        c.decide(nav_observation(3.1,y=400),3.1)
        self.assertEqual(c.decide(nav_observation(3.17,y=400),3.17).reason,'rope_approach')
        self.assertIn(('spawn','ground'),c.verified_edges)
        self.assertEqual(c.decide(nav_observation(4,x=420,y=400),4).keys,{'alt','up'})
        self.assertEqual(c.decide(nav_observation(4.5,x=420,y=330,vy=-120),4.5).reason,'rope_catch')
        c.decide(nav_observation(4.7,x=420,y=300,vy=-120),4.7)
        self.assertEqual(c.decide(nav_observation(4.84,x=420,y=285,vy=-120),4.84).reason,'rope_ascend')
        c.decide(nav_observation(5.5,x=440,y=210),5.5)
        self.assertEqual(c.decide(nav_observation(5.57,x=440,y=210),5.57).reason,'climb_complete')
        self.assertIn(('ground','combat'),c.verified_edges)
        monster=Actor(Box(550,165,585,210),.95)
        self.assertEqual(c.decide(nav_observation(5.61,x=440,y=210,monsters=[monster]),5.61).reason,'turn')
        self.assertEqual(c.decide(nav_observation(5.67,x=440,y=210,monsters=[monster]),5.67).keys,{'shift'})

    def test_brief_occlusion_releases_but_preserves_route(self):
        c=Controller(); c.last_epoch=1; c.transition=('spawn','ground','drop'); c.transition_started=1
        o=nav_observation(1.1); o.player=None; o.reason='player_not_found'
        self.assertFalse(c.decide(o,1.1).keys); self.assertIsNotNone(c.transition)
        o.captured_at=1.6; c.decide(o,1.6); self.assertIsNone(c.transition)

    def test_ascending_through_platform_is_not_landing(self):
        o=nav_observation(1,vy=-250)
        self.assertIsNone(standing_platform(o))

    def test_rope_target_is_exact_and_failure_replans(self):
        c=RopeClimber(target_id='missing')
        self.assertEqual(c.decide(nav_observation(1,x=420,y=400),1).reason,'no_accessible_rope')
        self.assertEqual(c.phase,'failed')

    def test_rope_reset_clears_completion(self):
        c=RopeClimber(); c.done=True; c.reset(); self.assertFalse(c.done)

    def test_rope_jump_waits_for_horizontal_momentum_to_stop(self):
        c=RopeClimber(target_id='combat')
        o=nav_observation(1,x=410,y=400); o.player=Actor(o.player.box,.99,vx=150)
        self.assertEqual(c.decide(o,1).reason,'rope_brake')
        d=c.decide(nav_observation(1.08,x=420,y=400),1.08)
        self.assertFalse(d.keys)
        self.assertEqual(c.decide(nav_observation(1.2,x=420,y=400),1.2).keys,{'up','alt'})

    def test_diagonal_rope_catch_holds_up_and_releases_lateral_at_rope(self):
        c=RopeClimber(target_id='combat',speed=180)
        self.assertEqual(c.decide(nav_observation(1,x=396,y=400),1).keys,{'right','up','alt'})
        self.assertEqual(c.decide(nav_observation(1.15,x=400,y=360,vy=-200),1.15).keys,{'right','up'})
        self.assertEqual(c.decide(nav_observation(1.24,x=419,y=340,vy=-200),1.24).keys,{'up'})
        self.assertEqual(c.decide(nav_observation(1.50,x=420,y=310,vy=-100),1.50).keys,{'up'})

    def test_no_downward_rope_edge_for_upward_only_climber(self):
        ps=[Platform('a',0,640,100),Platform('b',0,640,400)]
        self.assertNotIn(('b','rope'),graph(ps,[Rope(300,100,350)],MotionProfile(220,80,160,True))['a'])

    def test_launch_timeout_does_not_walk_forever(self):
        c=Controller(MotionProfile(220,80,160,True),True); c.last_epoch=1
        c.pending_edge=('ground','combat','jump'); c.pending_since=1
        self.assertEqual(c.decide(nav_observation(5.1,x=100,y=400),5.1).reason,'launch_approach_timeout')
        self.assertIn(('ground','combat'),c.failures)

    def test_unexpected_landing_releases_and_replans(self):
        c=Controller(); c.last_epoch=1; c.transition=('spawn','combat','jump'); c.transition_started=1
        d=c.decide(nav_observation(1.5,y=400),1.5)
        self.assertEqual(d.reason,'landed_elsewhere_replan'); self.assertFalse(d.keys)
        self.assertIsNone(c.transition)
