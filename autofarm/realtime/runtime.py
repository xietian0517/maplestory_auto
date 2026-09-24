"""Observable local runtime. Dry-run by default; live input needs explicit mode."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import json
from pathlib import Path
import time

import cv2
import numpy as np

from .control import Controller, LeasedKeys
from .model import Decision, MotionProfile
from .perception import GroundedVision, LatestCapture
from .semantic import atomic_json, load_scene, make_request, OpenAIPlanner
from .evidence import EvidenceRecorder,InputSessionLock


class Metrics:
    def __init__(self): self.rows=[]; self.started=time.perf_counter()
    def add(self,packet,finished,decision,reason,applied,perception_ms):
        self.rows.append(dict(t=finished-self.started,frame_id=packet.id,
                             capture_ms=(packet.finished-packet.started)*1000,
                             frame_to_decision_ms=(finished-packet.started)*1000,
                             perception_ms=perception_ms,reason=reason or decision.reason,
                             keys=sorted(decision.keys),input_applied=applied))
    def summary(self):
        elapsed=max(.001,time.perf_counter()-self.started)
        values=np.array([x['frame_to_decision_ms'] for x in self.rows])
        gaps=np.diff([x['t'] for x in self.rows])*1000
        valid=[x for x in self.rows if x['reason'] not in {'stale_frame','camera_or_map_changed','player_not_found','identity_confirming','focus_lost','waiting_for_gpt','window_resized'}]
        def stats(a):
            return dict(p50=round(float(np.percentile(a,50)),2),p95=round(float(np.percentile(a,95)),2),
                        p99=round(float(np.percentile(a,99)),2),maximum=round(float(max(a)),2)) if len(a) else None
        bins=np.bincount([max(0,int(x['t'])) for x in valid],minlength=max(1,int(elapsed)+1))
        windows=np.convolve(bins,np.ones(10,dtype=int),'valid')[:max(0,int(elapsed)-9)]/10 if elapsed>=10 else []
        return dict(elapsed_seconds=round(elapsed,2),frames=len(self.rows),
                    decision_hz=round(len(self.rows)/elapsed,2),valid_decision_hz=round(len(valid)/elapsed,2),
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
        motion=None,online=False,model='gpt-6-astra',refresh_seconds=30,climb=False,record=False):
    if not 1<=seconds<=600 or not 12<=hz<=60: raise ValueError('seconds 1..600; hz 12..60')
    if live and adapter is None: raise ValueError('Live input requires a validated adapter')
    cv2.setNumThreads(2)
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
    if climb:
        from .climbing import RopeClimber
        controller=RopeClimber()
    elif navigate:
        from .calibration import NavigationController
        controller=NavigationController(motion)
    else: controller=Controller(motion,navigate)
    base=getattr(controller,"base",controller)
    if isinstance(base,Controller): base.prefer_jump_attacks=True
    from .combat import HasteRefresh
    haste=HasteRefresh()
    metrics=Metrics()
    from .recovery import ActiveRecovery
    recovery=ActiveRecovery()
    vision=None; scene_id=None; epoch=0; previous=0; saved=0; report_at=0; request_at=0
    scene_mtime=None; phase='running'; future=None; planner_error=None; capture_backend='initializing'
    error=None; recorder=None; released=False; final_reason='not_started'
    executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='semantic') if online else None
    try:
        if (folder/'scene.json').exists():
            scene,seed=load_scene(folder)
            if scene.confidence>=.75:
                vision=GroundedVision(scene,seed,async_reacquire=True); scene_id=scene.request_id; controller.preferred=scene.preferred
                restore_appearance(vision,folder)
                scene_mtime=(folder/'scene.json').stat().st_mtime_ns
        context=LeasedKeys(adapter,hwnd) if live else nullcontext(None)
        with (InputSessionLock() if live else nullcontext()), context as keys, \
                EvidenceRecorder(folder,limit=160 if record else 0) as recorder, \
                LatestCapture(api,hwnd,hz,lambda:keys.epoch if keys else 0) as capture:
            if record and vision: recorder.bind_scene(vision.scene,vision.seed)
            end=time.perf_counter()+seconds; initial_region=None; invalid_since=None; unknown_floor_since=None
            floor_request_pending=False
            while time.perf_counter()<end:
                if keys and keys.stopped.is_set():
                    phase='input_error' if keys.error else 'F11_stop'; error=keys.error
                    break
                if (folder/'STOP').exists(): phase='file_stop'; break
                packet=capture.next(previous)
                if packet is None:
                    if capture.error: raise RuntimeError(capture.error)
                    continue
                previous=packet.id; now=time.perf_counter()
                capture_backend=capture.backend
                if initial_region is None: initial_region=packet.region
                input_epoch=packet.input_epoch
                if not packet.foreground or api.get_foreground()!=hwnd:
                    if keys: keys.clear()
                    controller.reset()
                    recovery.reset()
                    metrics.add(packet,now,Decision(),'focus_lost',False,0)
                    if now-report_at>=1:
                        atomic_json(folder/'status.json',dict(mode='LIVE' if live else 'DRY_RUN',phase='running',
                            reason='focus_lost',keys=[],scene_id=scene_id,map_epoch=epoch,
                            capture_backend=capture_backend,player_visible=False,monsters=0,**metrics.summary()))
                        report_at=now
                    continue
                if packet.region[2:]!=initial_region[2:]:
                    if vision: vision.close()
                    vision=None; epoch+=1; controller.reset(); initial_region=packet.region
                scene_path=folder/'scene.json'
                if scene_path.exists() and scene_path.stat().st_mtime_ns!=scene_mtime:
                    try:
                        scene,seed=load_scene(folder)
                        if scene.confidence<.75: raise ValueError('GPT confidence below threshold')
                        previous_vision=vision
                        updated=GroundedVision(scene,seed,async_reacquire=True)
                        restore_appearance(updated,folder,previous_vision)
                        if previous_vision: previous_vision.close()
                        vision=updated; controller.reset(); controller.preferred=scene.preferred
                        scene_id=scene.request_id; scene_mtime=scene_path.stat().st_mtime_ns
                        if record: recorder.bind_scene(scene,seed)
                        epoch+=1; invalid_since=None
                        unknown_floor_since=None
                        floor_request_pending=False
                    except (ValueError,KeyError,OSError) as e:
                        planner_error=type(e).__name__+': '+str(e)
                        scene_mtime=scene_path.stat().st_mtime_ns
                if future and future.done():
                    try: future.result(); planner_error=None
                    except Exception as e: planner_error=type(e).__name__+': '+str(e)
                    future=None
                if (vision is None and now-request_at>5) or (online and now-request_at>refresh_seconds):
                    if future is None:
                        make_request(packet.image,folder,epoch=epoch); request_at=now
                        if online and planner_error is None:
                            future=executor.submit(OpenAIPlanner(model).plan,folder)
                        elif not online:
                            # External planning must not be invalidated every five seconds.
                            request_at=now+seconds
                start=time.perf_counter()
                o=vision.observe(packet,epoch) if vision else None
                perception_ms=(time.perf_counter()-start)*1000
                now=time.perf_counter()
                if keys:
                    for key,released_at,token in keys.pop_releases():
                        jump=getattr(base,'jump_combat',None)
                        if key=='alt' and jump and jump.started==token:
                            jump.jump_released_at=released_at
                decision=controller.decide(o,now) if o else Decision(reason='waiting_for_gpt')
                if now-packet.started>=.085: decision=Decision(reason='stale_frame')
                if o and o.reason=='camera_or_map_changed':
                    if invalid_since is None: invalid_since=now
                    if now-invalid_since>.3:
                        vision.close()
                        vision=None; epoch+=1; controller.reset(); request_at=0
                elif o and not o.reason: invalid_since=None
                if (o and o.player and not o.reason and abs(o.player.vy)<60
                        and decision.reason in ('airborne_or_floor_unknown','climb_wait_for_floor','approach_wait_for_floor')):
                    if unknown_floor_since is None: unknown_floor_since=now
                    if now-unknown_floor_since>1.25 and not floor_request_pending and future is None:
                        make_request(packet.image,folder,epoch=epoch); request_at=now
                        floor_request_pending=True
                        if online and planner_error is None:
                            future=executor.submit(OpenAIPlanner(model).plan,folder)
                else: unknown_floor_since=None
                if navigate and not climb:
                    decision=recovery.apply(o,decision,now,packet.image.shape[1],vision.offset if vision else (0,0),controller)
                if hasattr(base,"orient_attack"): decision=base.orient_attack(o,decision,now)
                decision=haste.apply(o,decision,now,controller)
                applied=False
                if keys:
                    jump=getattr(base,'jump_combat',None)
                    pulses={'alt':(jump.EARLY_JUMP_HOLD,jump.started)} if (
                        jump and decision.reason=='jump_attack_takeoff' and getattr(jump,'early_jump',False)) else None
                    applied=keys.apply(decision.keys,packet.started,input_epoch,pulses=pulses)
                if applied or not live:
                    if hasattr(base,"acknowledge"): base.acknowledge(decision,now)
                    haste.on_input_applied(decision,now)
                finished=time.perf_counter()
                final_reason=decision.reason
                metrics.add(packet,finished,decision,o.reason if o else 'waiting_for_gpt',applied,perception_ms)
                if o and o.player and vision:
                    metrics.rows[-1]['player']=[round(o.player.box.cx,2),round(o.player.box.y2,2)]
                    metrics.rows[-1]['camera_offset']=list(vision.offset)
                    metrics.rows[-1]['player_velocity']=[o.player.vx,o.player.vy]
                    metrics.rows[-1]['identity_source']=vision.identity_source
                    metrics.rows[-1]['identity_confidence']=round(o.player.confidence,4)
                if applied and decision.reason in ('jump_attack_fire_first','jump_attack_fire_second'):
                    jump=getattr(base,'jump_combat',None)
                    if jump and jump.jumped_at is not None:
                        metrics.rows[-1]['jump_to_attack_ms']=round((finished-jump.jumped_at)*1000,2)
                        metrics.rows[-1]['jump_attack_mode']=getattr(jump,'fire_mode','unknown')
                metrics.rows[-1]['target']=decision.target
                metrics.rows[-1]['applied_facing']=getattr(base,'applied_facing',None)
                metrics.rows[-1]['scene_id']=scene_id
                metrics.rows[-1]['monsters']=[list(vars(m.box).values()) for m in o.monsters] if o else []
                metrics.rows[-1]['navigation_targets']=[list(vars(m.box).values()) for m in o.navigation_targets] if o else []
                if record and vision: recorder.offer(packet,metrics.rows[-1])
                if climb and decision.reason=='climb_complete':
                    phase='climb_complete'
                    if keys: keys.clear()
                if now-report_at>=1:
                    summary=metrics.summary()
                    atomic_json(folder/'status.json',dict(mode='LIVE' if live else 'DRY_RUN',phase=phase,
                        reason=decision.reason,keys=sorted(decision.keys),scene_id=scene_id,map_epoch=epoch,
                        capture_backend=capture_backend,model_connection='online' if online else 'external_conversation',planner_error=planner_error,
                        player_visible=bool(o and o.player),monsters=len(o.monsters) if o else 0,
                        navigation_targets=len(o.navigation_targets) if o else 0,**summary))
                    report_at=now
                if o and vision and now-saved>=2:
                    view=vision.annotate(packet.image,o,decision,metrics.summary()['decision_hz'],(finished-packet.started)*1000)
                    recorder.preview(view)
                    atomic_json(folder/'map.json',dict(map_name=vision.scene.map_name,epoch=epoch,
                        platforms=[vars(p) for p in o.platforms],ropes=[vars(r) for r in o.ropes],
                        verified_edges=list(controller.verified_edges),failed_edges=list(controller.failures)))
                    if getattr(controller,'result',None):
                        atomic_json(folder/'motion.json',dict(**vars(controller.result),
                            jump_distance_basis='conservative_kinematic_estimate; verify route edges in game'))
                    saved=now
                if phase=='climb_complete': break
        released=not live or not keys.held
    except Exception as exc:
        phase='error'; error=type(exc).__name__+': '+str(exc)
        raise
    finally:
        if vision: vision.close()
        if executor: executor.shutdown(wait=False,cancel_futures=True)
        summary=metrics.summary()
        if phase=='running': phase='completed'
        report=dict(mode='LIVE' if live else 'DRY_RUN',phase=phase,capture_backend=capture_backend,
                    reason=final_reason,keys=[],error=error,keys_released=released,
                    verified_edges=list(controller.verified_edges),failed_edges=list(controller.failures),
                    recording_frames=recorder.count if recorder else 0,
                    recording_error=recorder.error if recorder else None,**summary)
        atomic_json(folder/'report.json',report)
        atomic_json(folder/'status.json',dict(scene_id=scene_id,monsters=0,**report))
        with (folder/'frames.jsonl').open('w',encoding='utf-8') as f:
            for row in metrics.rows: f.write(json.dumps(row)+'\n')
    return summary
