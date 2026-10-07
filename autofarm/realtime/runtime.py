"""Observable local runtime. Dry-run by default; live input needs explicit mode."""
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import time

import cv2
import numpy as np

from .control import Controller, LeasedKeys, standing_platform
from .model import Decision, MotionProfile
from .perception import GroundedVision, LatestCapture
from .semantic import atomic_json, load_scene, make_request
from .evidence import EvidenceRecorder,InputSessionLock


class Metrics:
    def __init__(self,max_rows=None,journal=None):
        self.rows=deque(maxlen=max_rows) if max_rows else []
        self.started=time.perf_counter()
        self.journal=Path(journal).open('w',encoding='utf-8') if journal else None
        self.journal_flushed_at=self.started
        self.pending_row=None
    def flush_row(self):
        if self.journal and self.pending_row is not None:
            self.journal.write(json.dumps(self.pending_row)+'\n')
            self.pending_row=None
            now=time.perf_counter()
            if now-self.journal_flushed_at>=1:
                self.journal.flush();self.journal_flushed_at=now
    def close(self):
        self.flush_row()
        if self.journal:self.journal.close();self.journal=None
    def add(self,packet,finished,decision,reason,applied,perception_ms):
        # The previous row has now received all observation/rope annotations.
        self.flush_row()
        self.rows.append(dict(t=finished-self.started,frame_id=packet.id,
                             capture_ms=(packet.finished-packet.started)*1000,
                             frame_to_decision_ms=(finished-packet.started)*1000,
                             perception_ms=perception_ms,reason=reason or decision.reason,
                             keys=sorted(decision.keys),input_applied=applied))
        self.pending_row=self.rows[-1]
    def summary(self):
        elapsed=max(.001,time.perf_counter()-self.started)
        offset=self.rows[0]['t'] if isinstance(self.rows,deque) and len(self.rows)==self.rows.maxlen else 0
        window_elapsed=max(.001,elapsed-offset)
        values=np.array([x['frame_to_decision_ms'] for x in self.rows])
        gaps=np.diff([x['t'] for x in self.rows])*1000
        valid=[x for x in self.rows if x['reason'] not in {'stale_frame','camera_or_map_changed','player_not_found','identity_confirming','focus_lost','waiting_for_gpt','window_resized'}]
        def stats(a):
            return dict(p50=round(float(np.percentile(a,50)),2),p95=round(float(np.percentile(a,95)),2),
                        p99=round(float(np.percentile(a,99)),2),maximum=round(float(max(a)),2)) if len(a) else None
        bins=np.bincount([max(0,int(x['t']-offset)) for x in valid],minlength=max(1,int(window_elapsed)+1))
        windows=np.convolve(bins,np.ones(10,dtype=int),'valid')[:max(0,int(window_elapsed)-9)]/10 if window_elapsed>=10 else []
        return dict(elapsed_seconds=round(elapsed,2),frames=len(self.rows),
                    metrics_window_seconds=round(window_elapsed,2),
                    decision_hz=round(len(self.rows)/window_elapsed,2),valid_decision_hz=round(len(valid)/window_elapsed,2),
                    minimum_10s_valid_hz=round(float(min(windows)),2) if len(windows) else None,
                    frame_to_decision_ms=stats(values),decision_gap_ms=stats(gaps),
                    over_100ms=int(sum(v>=100 for v in values)),
                    reasons=dict(Counter(x['reason'] for x in self.rows)),
                    input_updates=sum(x['input_applied'] for x in self.rows),
                    active_input_frames=sum(x['input_applied'] and bool(x['keys']) for x in self.rows),
                    attack_input_frames=sum(x['input_applied'] and 'shift' in x['keys'] for x in self.rows),
                    jump_attack_timing={mode:dict(count=len(v),median_ms=round(float(np.median(v)),2),
                        min_ms=round(min(v),2),max_ms=round(max(v),2))
                        for mode in ('early','early_recovery','upper')
                        if (v:=[r['jump_to_attack_ms'] for r in self.rows if r.get('jump_attack_mode')==mode])},
                    measurement='capture start to decision/input submission; NOT display-event or game-action latency')


def attack_hold_seconds(value,default=None):
    """User-configured floor on the attack key's down time. 0 disables it."""
    if value is None: return default
    if type(value) not in (int,float) or not (value==0 or LeasedKeys.HOLD_MIN<=value<=LeasedKeys.HOLD_MAX):
        raise ValueError('Attack hold must be 0 (off) or between %.2f and %.1f seconds'
                         %(LeasedKeys.HOLD_MIN,LeasedKeys.HOLD_MAX))
    return float(value)


def run_duration_seconds(value):
    """Shared GUI/CLI contract: zero is continuous, positive values are seconds."""
    if type(value) not in (int,float) or not math.isfinite(value) or value<0 or 0<value<1:
        raise ValueError('时长必须为0（不限时）或至少1秒')
    return float(value)


def restore_appearance(vision,folder,previous=None):
    """Retain explicitly supplied session poses across semantic refreshes."""
    poses=list(previous.appearance.poses) if previous else []
    path=Path(folder)/'identity_poses.npy'
    if not poses and path.exists():
        bank=np.load(path,allow_pickle=False)
        if bank.dtype!=np.uint8 or bank.ndim!=4 or bank.shape[1:]!=(65,52,3) or not 1<=len(bank)<=3:
            raise ValueError('Invalid session identity poses')
        poses=list(bank)
    if poses:
        vision.appearance.poses=(poses[:2]+vision.appearance.poses[-1:])[:3]
    monsters=Path(folder)/'monster_templates.npz'
    if hasattr(vision,'templates') and monsters.exists():
        from .perception import gray_small
        with np.load(monsters,allow_pickle=False) as bank:
            for key in bank.files[:3]:
                im=bank[key]
                if im.dtype!=np.uint8 or im.ndim!=3 or im.shape[2]!=3 or not (12<=min(im.shape[:2]) and max(im.shape[:2])<=180):
                    raise ValueError('Invalid monster appearance')
                for sample in (im,cv2.flip(im,1)):
                    vision.templates.append((sample.copy(),gray_small(sample)))


def run(api,hwnd,folder,seconds=60,hz=30,live=False,adapter=None,navigate=False,
        motion=None,online=False,model='gpt-6-astra',refresh_seconds=30,climb=False,record=False,minimap=True,evaluation=False,park_only=False,jump_attacks=False,hybrid_attacks=False,park_on_finish=False,attack_hold=None,
        policy_mode=None,policy_model=None,action_policy=None,policy_provider=None):
    seconds=run_duration_seconds(seconds)
    if not 12<=hz<=60: raise ValueError('hz must be between 12 and 60')
    if live and adapter is None: raise ValueError('Live input requires a validated adapter')
    if evaluation and not live: raise ValueError('Evaluation evidence requires live mode')
    if park_only and not live: raise ValueError('Parking requires live mode')
    if jump_attacks and hybrid_attacks:raise ValueError('Choose one combat policy')
    if not minimap:raise ValueError('Player localization requires the minimap yellow marker')
    parking_enabled=evaluation or park_only or (live and park_on_finish)
    cv2.setNumThreads(2)
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
    from .policy import load_policy_config,key_variable
    policy_config=load_policy_config(folder,policy_mode,policy_model,policy_provider)
    from .scene_api import create_scene_planner
    scene_model=policy_config.model if policy_config.provider=='deepseek' else model
    if online and not os.environ.get(key_variable(policy_config.provider)):
        raise ValueError('Online scene recognition requires '+key_variable(policy_config.provider))
    if climb and policy_config.mode!='rule':raise ValueError('Climb test requires the rule policy')
    if policy_config.mode!='rule' and action_policy is None and not os.environ.get(key_variable(policy_config.provider)):
        raise ValueError('AI action policy requires '+key_variable(policy_config.provider)+'; scene understanding is configured separately')
    if climb:
        from .climbing import RopeClimber
        controller=RopeClimber()
    elif navigate:
        from .calibration import NavigationController
        controller=NavigationController(motion)
    else: controller=Controller(motion,navigate)
    base=getattr(controller,"base",controller)
    # Explicit command line/GUI value; farm_plan.json may supply one when omitted.
    requested_attack_hold=attack_hold_seconds(attack_hold); planned_attack_hold=None
    if isinstance(base,Controller):
        base.direct_attacks=not (jump_attacks or hybrid_attacks);base.prefer_jump_attacks=jump_attacks
        # Jump attacks are off unless the caller explicitly asks for the
        # pure-jump policy. hybrid/standing keep the reviewed standing firing
        # path and never lift off, so a wounded survivor can no longer pull the
        # controller into jump_attack_align/brake loops.
        base.jump_attacks_enabled=bool(jump_attacks)
    atomic_json(folder/'run_configuration.json',dict(seconds=seconds,hz=hz,live=live,navigate=navigate,
        combat_policy='hybrid' if hybrid_attacks else 'observed_jump' if jump_attacks else 'standing',jump_attacks_enabled=bool(jump_attacks),evaluation=evaluation,park_only=park_only,
        initial_motion=vars(motion).copy() if motion else None))
    atomic_json(folder/'action_policy_configuration.json',policy_config.data())
    from .combat import HasteRefresh
    haste=HasteRefresh()
    metrics=Metrics(max_rows=18000 if seconds==0 or seconds>600 else None,journal=folder/'frames.jsonl')
    from .recovery import ActiveRecovery
    recovery=ActiveRecovery()
    vision=None; scene_id=None; epoch=0; previous=0; saved=0; report_at=0; request_at=0
    scene_stamps={}; phase='running'; future=None; planner_error=None; capture_backend='initializing'
    refresh_folder=folder/'refresh'
    error=None; recorder=None; released=False; final_reason='not_started'
    parker=None;parking_deadline=None;parking_scene=None;farming_summary=None
    attack_hold_applied=0.0
    health=None;health_return=None;policy_request=None
    pipeline=None;keys=None
    executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='semantic') if online else None
    try:
        if (folder/'scene.json').exists():
            scene,seed=load_scene(folder)
            if scene.confidence>=.75:
                name_path=folder/'identity_name.png'
                name_template=cv2.imread(str(name_path)) if name_path.exists() else None
                if name_path.exists() and name_template is None:raise ValueError('Invalid saved player name template')
                vision=GroundedVision(scene,seed,name_template=name_template,async_reacquire=True,minimap=minimap); scene_id=scene.request_id; controller.preferred=scene.preferred
                restore_appearance(vision,folder)
                from .world_geometry import restore_world
                restore_world(vision,folder)
                base.hud_boxes=list(scene.hud_boxes)
                scene_stamps[folder]=(folder/'scene.json').stat().st_mtime_ns
        if parking_enabled:
            from .parking import SafeParking,validate_parking
            if vision is None:raise ValueError('Live evaluation/parking requires a validated scene and refuge plan')
            validate_parking(json.loads((folder/'parking.json').read_text(encoding='utf-8')),vision.scene)
        if live and (folder/'health.json').exists():
            if not parking_enabled or vision is None:raise ValueError('Health return requires scene and parking plan')
            from .health import HealthGuard
            health=HealthGuard(json.loads((folder/'health.json').read_text(encoding='utf-8')),
                               vision.scene.request_id,vision.scene.width,vision.scene.height)
        if (folder/'navigation_policy.json').exists() and not park_only:
            from .demo_policy import PlatformIntent
            spec=json.loads((folder/'navigation_policy.json').read_text(encoding='utf-8'))
            if (vision is None or not navigate or climb or spec.get('request_id')!=vision.scene.request_id
                    or spec.get('mode') not in ('shadow','active') or not .5<=spec.get('minimum_vote',.55)<=1
                    or type(spec.get('allow_stay',False))!=bool
                    or type(spec.get('search_commit_seconds',0)) not in (int,float)
                    or not 0<=spec.get('search_commit_seconds',0)<=2):
                raise ValueError('Navigation policy requires matching scene, navigation mode and valid vote threshold')
            base.platform_policy=PlatformIntent(json.loads((folder/'platform_intent.json').read_text(encoding='utf-8')))
            base.policy_mode=spec['mode'];base.policy_min_vote=spec.get('minimum_vote',.55)
            base.policy_allow_stay=spec.get('allow_stay',False)
            base.policy_search_seconds=spec.get('search_commit_seconds',0)
            policy_request=spec['request_id']
        from .live_evaluation import LiveEvaluation
        anchor_request=None
        if (folder/'farm_plan.json').exists() and not park_only:
            spec=json.loads((folder/'farm_plan.json').read_text(encoding='utf-8'))
            if (vision is None or not navigate or climb or spec.get('request_id')!=vision.scene.request_id
                    or spec.get('anchor') not in {p.id for p in vision.world_platforms}
                    or type(spec.get('quiet_seconds',6)) not in (int,float)
                    or not 0<spec.get('quiet_seconds',6)<=10):
                raise ValueError('Farm plan requires matching reviewed scene and bounded quiet wait')
            base.farm_anchor=spec['anchor'];base.anchor_quiet_seconds=spec.get('quiet_seconds',6)
            burst=spec.get('standing_burst_seconds',.5)
            if type(burst) not in (int,float) or not .5<=burst<=2:
                raise ValueError('Standing burst must be between 0.5 and 2 seconds')
            for flag in ('continue_short_occlusion','prefer_firing_anchor','refresh_attack_facing','sustain_farming'):
                if type(spec.get(flag,False)) is not bool:raise ValueError('Invalid farm plan flag: '+flag)
            base.standing_burst_seconds=burst
            base.standing_continue_occluded=spec.get('continue_short_occlusion',False)
            base.prefer_firing_anchor=spec.get('prefer_firing_anchor',False)
            base.refresh_attack_facing=spec.get('refresh_attack_facing',False)
            base.sustain_farming=spec.get('sustain_farming',False)
            if 'safe_platforms' in spec:
                safe=spec['safe_platforms'];firing=spec.get('firing_platforms',safe)
                ids={p.id for p in vision.world_platforms}
                if (not isinstance(safe,list) or not safe or not all(isinstance(x,str) and x in ids for x in safe)
                    or not isinstance(firing,list) or not firing or not all(x in safe for x in firing)):
                    raise ValueError('Safe firing plan requires reviewed platform IDs')
                base.safe_platforms=set(safe);base.firing_platforms=set(firing)
                base.sustain_farming=True
                lanes=spec.get('firing_lanes',{})
                if (not isinstance(lanes,dict) or any(k not in firing or not isinstance(v,dict)
                        or v.get('direction') not in ('left','right') or v.get('target') not in ids
                        for k,v in lanes.items()) or (lanes and set(lanes)!=set(firing))):
                    raise ValueError('Firing lanes require a direction and mapped monster platform for each perch')
                base.firing_lanes=lanes
                combat=spec.get('combat_platforms')
                if combat is not None:
                    if (not isinstance(combat,list) or not combat
                            or not all(isinstance(x,str) and x in ids and x not in safe for x in combat)):
                        raise ValueError('Combat platforms must be mapped monster floors separate from safe perches')
                    base.combat_platforms=set(combat)
            base.firing_viewport=(vision.scene.width,vision.scene.play_area.y2)
            anchor_request=spec['request_id']
            # A reviewed preset may carry its own attack hold; an explicit
            # command line/GUI value still wins over it.
            planned_attack_hold=attack_hold_seconds(spec.get('attack_hold_seconds'),None)
        if isinstance(base,Controller):
            base.attack_hold_seconds=(requested_attack_hold if requested_attack_hold is not None else
                                      planned_attack_hold if planned_attack_hold is not None else
                                      base.attack_hold_seconds)
        attack_hold_applied=float(getattr(base,'attack_hold_seconds',0) or 0)
        from .pipeline import DecisionPipeline
        pipeline=DecisionPipeline(controller,policy_config,navigate,climb,folder,action_policy,
                                  recovery=recovery,haste=haste)
        evidence=LiveEvaluation(folder,adapter,hz,hwnd=hwnd) if evaluation else None
        context=LeasedKeys(evidence or adapter,hwnd) if live else nullcontext(None)
        with (InputSessionLock() if live else nullcontext()), (evidence or nullcontext()), context as keys, \
                EvidenceRecorder(folder,limit=160 if record else 0) as recorder, \
                LatestCapture(api,hwnd,hz,lambda:keys.epoch if keys else 0) as capture:
            if record and vision: recorder.bind_scene(vision.scene,vision.seed,vision.world_data,vision.minimap_calibration_data())
            end=time.perf_counter()+seconds if seconds else float('inf')
            initial_region=None; invalid_since=None; unknown_floor_since=None
            floor_request_pending=False
            while True:
                if keys and keys.stopped.is_set():
                    phase='input_error' if keys.error else 'F11_stop'; error=keys.error
                    break
                if (folder/'STOP').exists(): phase='file_stop'; break
                if parker is None and (park_only or health_return or time.perf_counter()>=end or (folder/'PARK').exists()):
                    if not parking_enabled:break
                    if keys:keys.clear()
                    farming_summary=metrics.summary()
                    plan=json.loads((folder/'parking.json').read_text(encoding='utf-8'))
                    ids,roi=validate_parking(plan,vision.scene)
                    parker=SafeParking(getattr(base,'motion',motion or MotionProfile()),ids,roi)
                    base=parker.base;parking_scene=vision.scene.request_id
                    base.hud_boxes=list(vision.scene.hud_boxes)
                    parking_deadline=time.perf_counter()+180;phase='parking'
                    if evidence:evidence.mark_farming_end()
                if parker and time.perf_counter()>=parking_deadline:
                    phase='parking_unconfirmed';final_reason='parking_time_limit';break
                packet=capture.next(previous)
                if packet is None:
                    if capture.error: raise RuntimeError(capture.error)
                    continue
                previous=packet.id; now=time.perf_counter()
                if evidence:evidence.offer(packet)
                capture_backend=capture.backend
                if initial_region is None: initial_region=packet.region
                input_epoch=packet.input_epoch
                if not packet.foreground or api.get_foreground()!=hwnd:
                    if keys: keys.clear()
                    pipeline.reset(now,'focus_lost')
                    focus_decision=pipeline.step(None,now,scene_id=scene_id or '',
                        override=Decision(reason='focus_lost'),override_source='focus_guard')
                    pipeline.commit(focus_decision,False,now)
                    metrics.add(packet,now,Decision(),'focus_lost',False,0)
                    if now-report_at>=1:
                        atomic_json(folder/'status.json',dict(mode='LIVE' if live else 'DRY_RUN',phase='running',
                            reason='focus_lost',keys=[],scene_id=scene_id,map_epoch=epoch,
                            capture_backend=capture_backend,player_visible=False,monsters=0,**metrics.summary()))
                        report_at=now
                    continue
                if packet.region[2:]!=initial_region[2:]:
                    if vision: vision.close()
                    vision=None; epoch+=1; pipeline.reset(now,'window_resized'); initial_region=packet.region
                if health and not parker:
                    health_return=health.observe(packet.image,packet.started)
                    if health_return:
                        if keys:keys.clear()
                        health_decision=pipeline.step(None,now,scene_id=scene_id or '',health=health.status(),
                            override=Decision(reason=health_return),override_source='health_guard')
                        pipeline.commit(health_decision,False,now)
                        atomic_json(folder/'health_return.json',dict(reason=health_return,
                            t=now-metrics.started,frame_id=packet.id,**health.status()))
                        if health_return=='health_depleted_stop':
                            phase='health_stop';final_reason=health_return
                            if evidence:evidence.mark_farming_end()
                            break
                        # Next loop enters the existing parking state machine;
                        # no combat/navigation input is sent on this frame.
                        continue
                    if health.return_enabled and health.hp is None or health.low_samples:
                        if keys:keys.clear()
                        health_decision=pipeline.step(None,now,scene_id=scene_id or '',health=health.status(),
                            override=Decision(reason='health_confirming'),override_source='health_guard')
                        pipeline.commit(health_decision,False,now)
                        metrics.add(packet,now,Decision(),'health_confirming',False,0)
                        metrics.rows[-1]['phase']='farming'
                        if now-report_at>=1:
                            atomic_json(folder/'status.json',dict(mode='LIVE',phase=phase,
                                reason='health_confirming',keys=[],health=health.status(),**metrics.summary()))
                            report_at=now
                        continue
                for candidate_folder in (folder,refresh_folder):
                    scene_path=candidate_folder/'scene.json'
                    if not scene_path.exists() or scene_path.stat().st_mtime_ns==scene_stamps.get(candidate_folder): continue
                    try:
                        scene,seed=load_scene(candidate_folder)
                        if scene.confidence<.75: raise ValueError('GPT confidence below threshold')
                        previous_vision=vision
                        name_template=(previous_vision.name if previous_vision is not None
                            and previous_vision.name.shape[:2]==(round(scene.name_box.height),round(scene.name_box.width)) else None)
                        updated=GroundedVision(scene,seed,name_template=name_template,async_reacquire=True,minimap=minimap)
                        restore_appearance(updated,folder,previous_vision)
                        from .world_geometry import restore_world
                        restore_world(updated,candidate_folder)
                        if previous_vision: previous_vision.close()
                        vision=updated; pipeline.reset(now,'semantic_refresh'); controller.preferred=scene.preferred
                        base.hud_boxes=list(scene.hud_boxes)
                        scene_id=scene.request_id; scene_stamps[candidate_folder]=scene_path.stat().st_mtime_ns
                        if policy_request and policy_request!=scene_id:base.platform_policy=None
                        if anchor_request and anchor_request!=scene_id:
                            base.farm_anchor=None
                            if base.safe_platforms is not None:
                                base.safe_platforms=set();base.firing_platforms=set()
                        if record: recorder.bind_scene(scene,seed,vision.world_data,vision.minimap_calibration_data())
                        epoch+=1; invalid_since=None
                        unknown_floor_since=None
                        floor_request_pending=False
                    except (ValueError,KeyError,OSError) as e:
                        planner_error=type(e).__name__+': '+str(e)
                        scene_stamps[candidate_folder]=scene_path.stat().st_mtime_ns
                if future and future.done():
                    try: future.result(); planner_error=None
                    except Exception as e: planner_error=type(e).__name__+': '+str(e)
                    future=None
                if (vision is None and now-request_at>5) or (online and now-request_at>refresh_seconds):
                    if future is None:
                        planning_folder=refresh_folder if scene_id else folder
                        make_request(packet.image,planning_folder,epoch=epoch); request_at=now
                        if online and planner_error is None:
                            future=executor.submit(create_scene_planner(policy_config.provider,scene_model).plan,planning_folder)
                        elif not online:
                            # External planning must not be invalidated every five seconds.
                            request_at=now+seconds if seconds else float('inf')
                start=time.perf_counter()
                o=vision.observe(packet,epoch) if vision else None
                perception_ms=(time.perf_counter()-start)*1000
                now=time.perf_counter()
                input_releases=keys.pop_releases() if keys else []
                if parker and vision and parking_scene!=vision.scene.request_id:
                    plan=json.loads((folder/'parking.json').read_text(encoding='utf-8'))
                    ids,roi=validate_parking(plan,vision.scene)
                    parker=SafeParking(base.motion,ids,roi);base=parker.base;parking_scene=vision.scene.request_id
                parking_decision=parker.decide(o,now,packet.image) if parker else None
                if parking_decision is not None and hasattr(base,'orient_attack'):
                    parking_decision=base.orient_attack(o,parking_decision,now)
                if o and o.reason=='camera_or_map_changed':
                    if invalid_since is None: invalid_since=now
                    # Camera loss is not proof of a new map. Keep trying the
                    # existing scene, and preserve its loadable request/seed.
                    if now-invalid_since>2 and not floor_request_pending and future is None:
                        make_request(packet.image,refresh_folder,epoch=epoch); request_at=now
                        floor_request_pending=True
                        if online and planner_error is None:
                            future=executor.submit(create_scene_planner(policy_config.provider,scene_model).plan,refresh_folder)
                elif o and not o.reason:
                    if invalid_since is not None:
                        pipeline.reset(now,'camera_recovered')
                    invalid_since=None
                # Run the same pipeline used by recorded-state replay.
                decision=pipeline.step(o,now,packet.image,vision.offset if vision else (0,0),scene_id or '',
                    health.status() if health else None,override=parking_decision,
                    override_source='parking' if parker else None,releases=input_releases)
                if now-packet.started>=.085: decision=Decision(reason='stale_frame')
                if (o and o.player and not o.reason and abs(o.player.vy)<60
                        and decision.reason in ('airborne_or_floor_unknown','climb_wait_for_floor','approach_wait_for_floor')):
                    if unknown_floor_since is None: unknown_floor_since=now
                    if now-unknown_floor_since>1.25 and not floor_request_pending and future is None:
                        make_request(packet.image,refresh_folder,epoch=epoch); request_at=now
                        floor_request_pending=True
                        if online and planner_error is None:
                            future=executor.submit(create_scene_planner(policy_config.provider,scene_model).plan,refresh_folder)
                else: unknown_floor_since=None
                applied=False
                if keys:
                    if o is None or o.reason or not o.player or not o.motion_valid:
                        # Losing the yellow marker revokes even a minimum attack
                        # hold; no remembered body/name position can authorize it.
                        keys.clear()
                    jump=getattr(base,'jump_combat',None)
                    pulses={'alt':(jump.EARLY_JUMP_HOLD,jump.started)} if (
                        jump and decision.reason=='jump_attack_takeoff' and getattr(jump,'early_jump',False)) else None
                    # A committed attack owns the key for a minimum time so a late
                    # or missing frame cannot cut it into a single instant press.
                    holds={'shift':attack_hold_applied} if attack_hold_applied>0 and 'shift' in decision.keys else None
                    applied=keys.apply(decision.keys,packet.started,input_epoch,pulses=pulses,holds=holds)
                if keys:
                    with keys.lock:actual_held=sorted(keys.held)
                else:actual_held=[]
                pipeline.commit(decision,applied,now,o,actual_held,simulated=not live,owner=base if parker else None)
                finished=time.perf_counter()
                final_reason=decision.reason
                metrics.add(packet,finished,decision,o.reason if o else 'waiting_for_gpt',applied,perception_ms)
                metrics.rows[-1]['phase']='parking' if parker else 'farming'
                decision_base=parker.base if parker else base
                if o and o.player and vision:
                    metrics.rows[-1]['player']=[round(o.player.box.cx,2),round(o.player.box.y2,2)]
                    metrics.rows[-1]['camera_offset']=list(vision.offset)
                    metrics.rows[-1]['player_velocity']=[o.player.vx,o.player.vy]
                    metrics.rows[-1]['identity_source']=vision.identity_source
                    metrics.rows[-1]['identity_confidence']=round(o.player.confidence,4)
                    floor=standing_platform(o)
                    metrics.rows[-1]['observed_floor']=vars(floor) if floor else None
                    jump=getattr(decision_base,'jump_combat',None)
                    if jump:
                        source=next((p for p in o.platforms if p.id==jump.floor_id),None)
                        metrics.rows[-1]['jump_evidence']=dict(source_floor=jump.floor_id,
                            rise=None if source is None else source.y-o.player.box.y2,
                            jump_age_ms=None if jump.jumped_at is None else (finished-jump.jumped_at)*1000,
                            shots_submitted=jump.shots_issued,target_id=jump.target_id)
                if applied and decision.reason in ('jump_attack_fire_first','jump_attack_fire_second'):
                    jump=getattr(base,'jump_combat',None)
                    if jump and jump.jumped_at is not None:
                        metrics.rows[-1]['jump_to_attack_ms']=round((finished-jump.jumped_at)*1000,2)
                        metrics.rows[-1]['jump_attack_mode']=getattr(jump,'fire_mode','unknown')
                metrics.rows[-1]['target']=decision.target
                metrics.rows[-1]['action_policy']=pipeline.status()
                metrics.rows[-1]['feedback']=pipeline.feedback.status(now)
                metrics.rows[-1]['platform_policy']=getattr(decision_base,'policy_advice',None)
                metrics.rows[-1]['firing_position']=getattr(decision_base,'firing_diagnostics',None)
                metrics.rows[-1]['applied_facing']=getattr(decision_base,'applied_facing',None)
                metrics.rows[-1]['navigation_edge']=(getattr(decision_base,'transition',None)
                    or getattr(decision_base,'pending_edge',None) or getattr(decision_base,'rope_edge',None))
                metrics.rows[-1]['scene_id']=scene_id
                metrics.rows[-1]['monsters']=[list(vars(m.box).values()) for m in o.monsters] if o else []
                metrics.rows[-1]['monster_tracks']=[dict(id=m.track_id,box=list(vars(m.box).values()),
                    confidence=float(m.confidence),vx=float(m.vx),vy=float(m.vy)) for m in o.monsters] if o else []
                metrics.rows[-1]['navigation_targets']=[list(vars(m.box).values()) for m in o.navigation_targets] if o else []
                metrics.rows[-1]['minimap']=vision.minimap.status() if vision else None
                metrics.rows[-1]['minimap_goals']=o.minimap_goals if o else []
                rope_controller=controller if climb else getattr(decision_base,'rope_climber',None)
                rope_status=(dict(phase=rope_controller.phase,attempts=rope_controller.attempts,
                                  target=rope_controller.target_id,grab_confirmed=rope_controller.grab_confirmed)
                             if rope_controller else None)
                metrics.rows[-1]['rope']=rope_status
                if record and vision: recorder.offer(packet,metrics.rows[-1])
                if climb and not parker and decision.reason=='climb_complete':
                    phase='climb_complete'
                    if keys: keys.clear()
                if now-report_at>=1:
                    summary=metrics.summary()
                    atomic_json(folder/'status.json',dict(mode='LIVE' if live else 'DRY_RUN',phase=phase,
                        reason=decision.reason,keys=sorted(decision.keys),scene_id=scene_id,map_epoch=epoch,
                        capture_backend=capture_backend,model_connection='online' if online else 'external_conversation',planner_error=planner_error,
                        refresh_folder=str(refresh_folder.resolve()) if floor_request_pending else None,
                        player_visible=bool(o and o.player),monsters=len(o.monsters) if o else 0,
                        navigation_targets=len(o.navigation_targets) if o else 0,
                        minimap=vision.minimap.status() if vision else None,rope=rope_status,
                        action_policy=pipeline.status(),feedback=pipeline.feedback.status(now),
                        health=health.status() if health else None,**summary))
                    report_at=now
                if o and vision and now-saved>=2:
                    view=vision.annotate(packet.image,o,decision,metrics.summary()['decision_hz'],(finished-packet.started)*1000)
                    recorder.preview(view)
                    atomic_json(folder/'map.json',dict(map_name=vision.scene.map_name,epoch=epoch,
                        platforms=[vars(p) for p in o.platforms],ropes=[vars(r) for r in o.ropes],
                        verified_edges=list(controller.verified_edges),failed_edges=list(controller.failures)))
                    atomic_json(folder/'minimap.json',vision.minimap.to_data())
                    if getattr(controller,'result',None):
                        atomic_json(folder/'motion.json',dict(**vars(controller.result),
                            jump_distance_basis='conservative_kinematic_estimate; verify route edges in game'))
                    saved=now
                if phase=='climb_complete':
                    if parking_enabled:
                        end=now;phase='running'
                    else:break
                if parker and parker.done:
                    if keys:keys.clear()
                    phase='parked';break
        released=not live or not keys.held
    except Exception as exc:
        phase='error'; error=type(exc).__name__+': '+str(exc)
        raise
    finally:
        # The input context has exited even when capture/serialization failed.
        # Report the actual released state rather than the pre-exit default.
        if keys is not None:released=not live or not keys.held
        if pipeline:pipeline.close(time.perf_counter())
        metrics.close()
        if vision: vision.close()
        if executor: executor.shutdown(wait=False,cancel_futures=True)
        summary=metrics.summary()
        if phase=='running': phase='completed'
        report=dict(mode='LIVE' if live else 'DRY_RUN',phase=phase,capture_backend=capture_backend,
                    reason=final_reason,keys=[],error=error,keys_released=released,
                    attack_hold_seconds=attack_hold_applied,
                    verified_edges=list(controller.verified_edges),failed_edges=list(controller.failures),
                    recording_frames=recorder.count if recorder else 0,
                    recording_error=recorder.error if recorder else None,
                    minimap=vision.minimap.status() if vision else None,**summary)
        report['parking']=parker.result() if parker else dict(confirmed=False,reason='not_performed')
        report['farming_summary']=farming_summary
        report['health']=health.status() if health else None
        report['action_policy']=pipeline.status() if pipeline else None
        atomic_json(folder/'report.json',report)
        atomic_json(folder/'status.json',dict(scene_id=scene_id,monsters=0,**report))
    return summary
