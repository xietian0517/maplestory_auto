"""Yellow-dot authority and safe-ledges-only route regressions."""
import json
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from autofarm.realtime.control import Controller, standing_platform
from autofarm.realtime.firing_position import choose_position, safe_transfer
from autofarm.realtime.minimap import locate_minimap, yellow_points
from autofarm.realtime.model import Actor, Box, Decision, MotionProfile, Observation, Platform, Scene
from autofarm.realtime.perception import Frame, GroundedVision, Camera
from autofarm.realtime.recovery import ActiveRecovery


def vision():
    root=Path(__file__).parent/'fixtures/realtime'
    data=json.loads((root/'scene.json').read_text(encoding='utf-8'))
    seed=cv2.imread(str(root/'seed.png'))
    scene=Scene.parse(data,data['request_id'],1366,768)
    v=GroundedVision(scene,seed)
    x,y,w,h=locate_minimap(seed)
    marker=np.array(yellow_points(seed[y:y+h,x:x+w])[0])
    foot=np.array([scene.name_box.cx,scene.name_box.cy+scene.foot_offset])
    v.minimap.configure(seed,dict(scale=.0625,intercept=(marker-foot*.0625).tolist()))
    for i in range(3):v.observe(Frame(i,1+i*.04,1+i*.04,seed,True,()))
    return v,seed


def test_sprite_and_name_cannot_override_yellow_dot():
    v,seed=vision()
    before=v.observe(Frame(4,1.12,1.12,seed,True,()))
    with patch.object(v,'_player',side_effect=AssertionError('sprite localization must not run')):
        hidden=seed.copy();hidden[340:480,600:740]=0
        after=v.observe(Frame(5,1.16,1.16,hidden,True,()))
    assert after.player is not None
    assert abs(after.player.box.cx-before.player.box.cx)<3
    assert v.identity_source=='minimap_yellow'


def test_missing_or_ambiguous_yellow_dot_never_falls_back():
    for ambiguous in (False,True):
        v,seed=vision();image=seed.copy();x,y,w,h=v.minimap.roi
        if ambiguous:cv2.circle(image,(x+15,y+65),2,(0,255,255),-1)
        else:
            pane=image[y:y+h,x:x+w]
            hsv=cv2.cvtColor(pane,cv2.COLOR_BGR2HSV)
            pane[cv2.inRange(hsv,(20,130,175),(38,255,255))>0]=(40,45,40)
        o=v.observe(Frame(4,1.12,1.12,image,True,()))
        assert o.player is None
        assert o.reason.startswith('minimap_')
        assert not Controller().decide(o,1.12).keys


def test_calibration_survives_pause_but_unrelated_terrain_does_not():
    v,seed=vision()
    for i in range(3):o=v.observe(Frame(i+5,3+i*.04,3+i*.04,seed,True,()))
    assert o.player is not None
    assert v.identity_source=='minimap_yellow'
    image=seed.copy();x,y,w,h=v.minimap.roi;image[y:y+h,x:x+w]=0
    o=v.observe(Frame(9,3.12,3.12,image,True,()))
    assert o.player is None


def test_clear_part_of_monster_floor_is_never_a_destination():
    floor=Platform('here',0,160,400);mobfloor=Platform('mob',180,700,400)
    o=Observation(1,1,Actor(Box(65,352,95,400),.99),
        [Actor(Box(450,352,480,400),.99)],platforms=[floor,mobfloor])
    choice=choose_position(o,floor,MotionProfile(180,110,220,True),
        safe_platforms={'here'},firing_platforms={'here'})
    assert choice[1]=='here' and choice[3]==[]


def test_route_cannot_use_monster_floor_as_intermediate():
    platforms=[Platform('here',0,100,200),Platform('mob',0,400,370),Platform('perch',0,100,550)]
    o=Observation(1,1,Actor(Box(35,152,65,200),.99),
        [Actor(Box(250,552,280,600),.99)],platforms=platforms)
    assert choose_position(o,platforms[0],MotionProfile(),
        safe_platforms={'here','perch'},firing_platforms={'perch'}) is None


def test_safe_wait_cannot_be_overridden_by_idle_recovery():
    c=Controller(navigate=True);c.sustain_farming=True;c.safe_platforms={'here'};c.firing_platforms={'here'}
    o=Observation(1,1,Actor(Box(85,252,115,300),.99),platforms=[Platform('here',0,200,300)])
    recovery=ActiveRecovery()
    for t in (1,2,8,30):
        o.captured_at=t;d=c.decide(o,t)
        assert not recovery.apply(o,d,t,1366,controller=c).keys
        assert c.transition is None and c.pending_edge is None


def test_offscreen_perch_uses_only_a_visible_safe_next_step():
    ps=[Platform('top',0,200,200),Platform('step',0,200,400),Platform('perch',0,200,600)]
    o=Observation(1,1,Actor(Box(85,152,115,200),.99),platforms=ps)
    path=safe_transfer(o,ps[0],MotionProfile(),{'top','step','perch'},{'perch'},viewport=(1366,500))
    assert path[0]==('top','step','drop')
    assert not safe_transfer(o,ps[0],MotionProfile(),{'top','perch'},{'perch'},viewport=(1366,500))


def test_shortest_offscreen_drop_does_not_hide_visible_detour():
    ps=[Platform('top',0,200,350),Platform('step',0,200,470),Platform('perch',0,200,588)]
    o=Observation(1,1,Actor(Box(85,302,115,350),.99),platforms=ps)
    path=safe_transfer(o,ps[0],MotionProfile(),{'top','step','perch'},{'perch'},viewport=(1366,570))
    assert path[0]==('top','step','drop')


def test_long_approach_holds_direction_without_periodic_stops():
    c=Controller(navigate=True)
    ps=[Platform('here',0,350,400),Platform('perch',250,450,460)]
    for t,x in ((1,50),(1.08,66),(1.3,98),(1.36,114)):
        o=Observation(t,1,Actor(Box(x-15,352,x+15,400),.99),platforms=ps,position_quantum=16)
        d=c._navigate_step(o,('here','perch','drop'),t)
        assert d.keys==frozenset({'right'}) and d.reason=='approach_launch'


def test_safe_policy_never_attacks_after_falling_onto_monster_floor():
    c=Controller();c.safe_platforms={'perch'}
    floor=Platform('monsters',0,800,400)
    o=Observation(1,1,Actor(Box(85,352,115,400),.99),
        [Actor(Box(250,352,280,400),.99)],platforms=[floor])
    assert c.fight(o,floor,1) is None


def test_minimap_resolution_recognizes_recorded_stationary_ledge_edge():
    o=Observation(1,1,Actor(Box(491.4,294.4,521.4,342.4),.99),
        platforms=[Platform('upper_left',519,646,351)],position_quantum=16)
    assert standing_platform(o).id=='upper_left'
    o.player=Actor(o.player.box,.99,vy=200)
    assert standing_platform(o) is None


def test_distributed_terrain_matches_relocalize_with_moving_sprite_occlusions():
    root=Path(__file__).parent/'fixtures/realtime'
    data=json.loads((root/'minimap_world_scene.json').read_text(encoding='utf-8'))
    scene=Scene.parse(data,data['request_id'],1366,768)
    seed=cv2.imread(str(root/'minimap_world_seed.png'))
    camera=Camera(seed,scene.play_area,scene.platforms,scene.ropes,scene.exclusions)
    image=cv2.imread(str(root/'minimap_relocalization.png'))
    np.testing.assert_allclose(camera.relocalize(image),(145,85),atol=2)


def test_annotated_arrow_excludes_opposite_and_unmarked_monster_platforms():
    c=Controller(navigate=True);c.safe_platforms={'perch'};c.firing_platforms={'perch'}
    c.firing_lanes={'perch':dict(direction='left',target='left_mobs')}
    ps=[Platform('perch',450,600,400),Platform('left_mobs',0,350,460),Platform('right_mobs',700,1000,460)]
    left=Actor(Box(250,412,280,460),.99);right=Actor(Box(750,412,780,460),.99)
    o=Observation(1,1,Actor(Box(485,352,515,400),.99),[left,right],platforms=ps)
    c.decide(o,1)
    assert c.standing_target(left,o.player.box)
    assert not c.standing_target(right,o.player.box)
    assert choose_position(o,ps[0],c.motion,safe_platforms={'perch'},firing_platforms={'perch'},
        firing_lanes=c.firing_lanes)[4]==1


def test_rope_top_accepts_one_marker_pixel_landing_error():
    from autofarm.realtime.climbing import RopeClimber
    from autofarm.realtime.model import Rope
    c=RopeClimber(target_id='top');c.target_id='top';c.attempts=1;c.phase='confirm';c.rope_index=0
    for t in (1,1.06):
        o=Observation(t,1,Actor(Box(85,341.4,115,389.4),.99),
            platforms=[Platform('top',0,200,400)],ropes=[Rope(100,400,650)],position_quantum=16)
        d=c.decide(o,t)
    assert c.done and d.reason=='climb_complete'


def test_user_marked_perches_exclude_bottom_and_have_exact_arrows():
    p=Path(__file__).resolve().parents[1]/'templates/monkey_forest_safe/farm_plan.json'
    d=json.loads(p.read_text(encoding='utf-8'))
    assert {'left_top','left_high','refuge_right','observed_17'}<set(d['firing_platforms'])
    assert set(d['firing_platforms'])<=set(d['safe_platforms'])
    assert not set(d['safe_platforms'])&{'observed_14','observed_15','observed_19','observed_20','observed_21','observed_22','reviewed_left_leaf'}
    assert {k:v['direction'] for k,v in d['reviewed_firing_examples'].items()}==dict(left_top='left',left_high='left',refuge_right='right',observed_17='left')
    assert not d.get('firing_lanes')
    assert not set(d['combat_platforms'])&set(d['safe_platforms'])


def test_adaptive_perch_attacks_either_side_but_never_pet_on_safe_step():
    c=Controller(navigate=True);c.safe_platforms={'perch','step'};c.firing_platforms={'perch'}
    c.combat_platforms={'left_mobs','right_mobs'}
    ps=[Platform('perch',450,600,400),Platform('left_mobs',0,300,460),
        Platform('right_mobs',750,1000,460),Platform('step',630,710,460)]
    left=Actor(Box(250,412,280,460),.99);right=Actor(Box(780,412,810,460),.99)
    pet=Actor(Box(650,412,680,460),.99)
    o=Observation(1,1,Actor(Box(485,352,515,400),.99),[left,right,pet],platforms=ps)
    c.decide(o,1)
    assert c.standing_target(left,o.player.box) and c.standing_target(right,o.player.box)
    assert not c.standing_target(pet,o.player.box)
    choice=choose_position(o,ps[0],c.motion,safe_platforms=c.safe_platforms,
        firing_platforms=c.firing_platforms,combat_platforms=c.combat_platforms)
    assert choice[1]=='perch' and choice[4]==2
    c.combat_platforms=set();c.decide(o,1)
    assert not c.standing_target(left,o.player.box)


def test_narrow_overlap_jumps_from_inside_ledge_instead_of_walking_to_edge():
    c=Controller(MotionProfile(160,113,109,True),navigate=True)
    ps=[Platform('source',427,554,708),Platform('dest',519,645,648)]
    o=Observation(1,1,Actor(Box(505,660,535,708),.99,vx=0),platforms=ps,position_quantum=16)
    d=c._navigate_step(o,('source','dest','jump'),1)
    assert d.reason=='jump_wait_takeoff' and d.keys==frozenset({'alt','right'})
    assert c.transition==('source','dest','jump')
    o.player=Actor(o.player.box.moved(0,-16),.99,vy=-100)
    d=c._navigate_step(o,('source','dest','jump'),1.1,airborne=True)
    assert d.reason=='jump' and d.keys==frozenset({'right'})


def test_safe_transfer_approach_does_not_replan_each_frame():
    c=Controller(navigate=True);c.sustain_farming=True;c.safe_platforms={'here','dest'}
    c.firing_platforms={'dest'};c.last_epoch=0
    c.pending_edge=('here','dest','jump');c.pending_since=1
    ps=[Platform('here',0,200,400),Platform('dest',150,350,340)]
    o=Observation(1,1,Actor(Box(35,352,65,400),.99),platforms=ps,position_quantum=16)
    with patch('autofarm.realtime.firing_position.choose_position',side_effect=AssertionError('committed approach must persist')):
        d=c.decide(o,1)
    assert d.reason=='approach_launch' and d.keys==frozenset({'right'})


def test_visible_monkeys_at_another_safe_perch_override_empty_perch_wait():
    c=Controller(navigate=True);c.sustain_farming=True;c.safe_platforms={'here','dest'}
    c.firing_platforms={'here','dest'};c.last_epoch=0
    ps=[Platform('here',0,200,300),Platform('dest',0,200,480)]
    o=Observation(1,1,Actor(Box(85,252,115,300),.99),platforms=ps,
        navigation_targets=[Actor(Box(300,482,330,530),.99)])
    d=c.decide(o,1)
    assert d.reason=='drop' and d.target=='dest'
    assert c.transition==('here','dest','drop')


def test_jump_apex_below_a_platform_is_not_a_landing():
    o=Observation(1,1,Actor(Box(491.4,550.4,521.4,598.4),.99),
        platforms=[Platform('left_high',427,556,588)],position_quantum=16)
    assert standing_platform(o) is None
    o.player=Actor(Box(491.4,534.4,521.4,582.4),.99)
    assert standing_platform(o).id=='left_high'


def test_jump_waits_for_committed_attack_to_finish():
    c=Controller(navigate=True);c.standing_last_applied=.9
    ps=[Platform('source',427,554,708),Platform('dest',519,645,648)]
    o=Observation(1,1,Actor(Box(505,646.4,535,694.4),.99),platforms=ps,position_quantum=16)
    assert c._navigate_step(o,('source','dest','jump'),1).reason=='attack_finish_before_move'
    assert c.transition is None
    assert 'alt' in c._navigate_step(o,('source','dest','jump'),1.3).keys


def test_rejected_jump_never_turns_into_walking_off_launch_platform():
    c=Controller(navigate=True)
    ps=[Platform('source',427,554,708),Platform('dest',519,645,648)]
    o=Observation(1,1,Actor(Box(505,646.4,535,694.4),.99),platforms=ps,position_quantum=16)
    edge=('source','dest','jump')
    for t in (1,1.04,1.12,1.3,1.36):
        d=c._navigate_step(o,edge,t,airborne=c.transition is not None)
        if t>=1.08:assert not d.keys&{'left','right'}
    assert d.reason=='jump_not_observed' and c.transition is None


def test_overlapping_perch_above_requires_vertical_takeoff_without_left_bias():
    c=Controller(navigate=True)
    ps=[Platform('source',339,467,770),Platform('dest',427,554,708)]
    o=Observation(1,1,Actor(Box(443.4,710.4,473.4,758.4),.99),platforms=ps,position_quantum=16)
    edge=('source','dest','jump')
    # Already stopped in the supported overlap after a launch brake.
    c.jump_brake_edge=edge;c.jump_brake_stable=(458.4-339,.5)
    d=c._navigate_step(o,edge,1)
    assert d.reason=='jump_wait_takeoff' and d.keys==frozenset({'alt'})
