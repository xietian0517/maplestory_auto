"""Extract observed demo states and human keys offline; never loads an input adapter."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from autofarm.realtime.control import standing_platform
from autofarm.realtime.model import Actor
from autofarm.realtime.perception import Frame,GroundedVision
from autofarm.realtime.runtime import restore_appearance
from autofarm.realtime.semantic import load_scene,atomic_json
from autofarm.realtime.world_geometry import restore_world


def extract(demo,scene_folder,out,hz=10,seconds=600):
    demo=Path(demo);scene_folder=Path(scene_folder);out=Path(out)
    out.mkdir(parents=True,exist_ok=False)
    cv2.setNumThreads(2)
    scene,seed=load_scene(scene_folder)
    name=cv2.imread(str(scene_folder/'identity_name.png'))
    vision=GroundedVision(scene,seed,name_template=name,async_reacquire=False,minimap=True)
    restore_appearance(vision,scene_folder);restore_world(vision,scene_folder)
    counts=Counter();minutes={};started=time.perf_counter();cap=None;segment=None;bucket=-1
    full_scan_at=-1;sample_count=0
    def actor(a):
        return dict(box=list(vars(a.box).values()),confidence=a.confidence,vx=a.vx,vy=a.vy,track_id=a.track_id)
    try:
        with (out/'states.jsonl').open('x',encoding='utf-8') as log:
            for line in (demo/'frames.jsonl').open(encoding='utf-8'):
                row=json.loads(line);t=row['capture_started']
                if t>seconds:break
                if row['segment']!=segment:
                    if cap:cap.release()
                    segment=row['segment'];cap=cv2.VideoCapture(str(demo/segment));index=0
                if row['video_frame']!=index:raise ValueError('Video index discontinuity')
                if not cap.grab():raise ValueError('Unreadable indexed frame')
                index+=1
                tick=int(t*hz)
                if tick==bucket:continue
                bucket=tick
                ok,image=cap.retrieve()
                if not ok:raise ValueError('Unreadable selected frame')
                o=vision.observe(Frame(row['frame_id'],t,t,image,row['foreground'],tuple(row['region'])))
                # Deterministic synchronous full-frame scan, retaining the same
                # 0.4s navigation-target cadence. This is offline analysis, not
                # a measurement of live pipeline latency or input success.
                if o.motion_valid and not o.reason and t-full_scan_at>=.4:
                    targets=vision.detect_monsters(image,scene.play_area)
                    dx,dy=vision.offset
                    vision.far_targets=[Actor(a.box.moved(-dx,-dy),a.confidence) for a in targets]
                    vision.far_targets_at=t;full_scan_at=t;o.navigation_targets=targets
                floor=standing_platform(o)
                reason=o.reason or ('grounded' if floor else 'airborne_or_unmapped')
                counts[reason]+=1;minute=minutes.setdefault(str(int(t//60)),Counter());minute[reason]+=1
                record=dict(t=t,frame_id=row['frame_id'],segment=segment,video_frame=row['video_frame'],
                    keys=row.get('keys'),key_sample_age=row.get('key_sample_age'),key_focus=row.get('key_focus'),
                    foreground=row['foreground'],reason=o.reason,identity_source=vision.identity_source,
                    camera_offset=list(vision.offset),player=actor(o.player) if o.player else None,
                    floor_id=floor.id if floor else None,monsters=[actor(a) for a in o.monsters],
                    navigation_targets=[actor(a) for a in o.navigation_targets],
                    platforms=[vars(p) for p in o.platforms],ropes=[vars(r) for r in o.ropes])
                log.write(json.dumps(record)+'\n');sample_count+=1
                if sample_count%100==0:
                    log.flush();atomic_json(out/'progress.json',dict(video_seconds=t,samples=sample_count,
                        wall_seconds=time.perf_counter()-started,counts=dict(counts)))
                    print(json.dumps(dict(video_seconds=round(t,1),samples=sample_count,counts=dict(counts))),flush=True)
        report=dict(mode='OFFLINE_HUMAN_STATE_EXTRACTION',samples=sample_count,hz=hz,
                    seconds_limit=seconds,wall_seconds=time.perf_counter()-started,counts=dict(counts),
                    by_minute={k:dict(v) for k,v in minutes.items()},automatic_inputs=False,
                    demo=str(demo.resolve()),scene=str(scene_folder.resolve()),
                    source_sha256={str(p.relative_to(Path(__file__).resolve().parents[1])):hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in (Path(__file__).resolve().parents[1]/'autofarm/realtime').glob('*.py')},
                    scene_sha256={n:hashlib.sha256((scene_folder/n).read_bytes()).hexdigest() for n in
                        ('request.json','scene.json','world_geometry.json','identity_poses.npy','identity_name.png','monster_templates.npz')},
                    limitation='Existing demonstration was already used for development. Labels are human inputs, not confirmed outcomes. No independent policy score.')
        atomic_json(out/'report.json',report)
        return report
    finally:
        vision.close()
        if cap:cap.release()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('demo');p.add_argument('scene');p.add_argument('output')
    p.add_argument('--hz',type=int,default=10,choices=(6,10,15,30));p.add_argument('--seconds',type=float,default=600)
    a=p.parse_args();print(json.dumps(extract(a.demo,a.scene,a.output,a.hz,a.seconds),indent=2))
