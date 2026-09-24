"""Replay recorded frames without importing Windows input or opening the game."""
import json
from pathlib import Path
import time

import cv2
import numpy as np

from .control import Controller
from .model import MotionProfile
from .perception import Frame,GroundedVision
from .semantic import atomic_json,load_scene


def replay(folder):
    folder=Path(folder); scene,seed=load_scene(folder)
    cv2.setNumThreads(2); vision=GroundedVision(scene,seed)
    motion_path=folder/'motion.json'
    motion=MotionProfile.parse(json.loads(motion_path.read_text(encoding='utf-8'))) if motion_path.exists() else None
    controller=Controller(motion,True); controller.preferred=scene.preferred
    records=[json.loads(line) for line in (folder/'replay'/'frames.jsonl').read_text(encoding='utf-8').splitlines() if line]
    rows=[]
    for r in records:
        if r.get('scene_id') and r['scene_id']!=scene.request_id:
            ident=r['scene_id']
            scene_folder=(folder/'replay'/'scenes'/ident).resolve()
            if scene_folder.parent!=(folder/'replay'/'scenes').resolve(): raise ValueError('Invalid recorded scene id')
            scene,seed=load_scene(scene_folder); vision=GroundedVision(scene,seed)
            controller=Controller(motion,True); controller.preferred=scene.preferred
        path=(folder/'replay'/r['image']).resolve()
        if path.parent!=(folder/'replay').resolve(): raise ValueError('Replay image outside recording')
        image=cv2.imread(str(path))
        if image is None: raise ValueError('Missing replay frame')
        stamp=r['captured_at']; packet=Frame(r['frame_id'],stamp,stamp,image,True,())
        start=time.perf_counter(); o=vision.observe(packet); d=controller.decide(o,stamp)
        elapsed=(time.perf_counter()-start)*1000
        rows.append(dict(frame_id=r['frame_id'],processing_ms=elapsed,reason=o.reason or d.reason,
                         player=[o.player.box.cx,o.player.box.y2] if o.player else None,
                         monsters=len(o.monsters),keys=sorted(d.keys)))
    times=[r['processing_ms'] for r in rows]
    report=dict(mode='OFFLINE_REPLAY',frames=len(rows),visible_frames=sum(r['player'] is not None for r in rows),
                processing_ms=dict(p50=float(np.percentile(times,50)),p99=float(np.percentile(times,99)),maximum=max(times)) if times else None,
                rows=rows,measurement='Saved frames only. No input sent; cannot prove game action success or live capture rate.')
    atomic_json(folder/'replay_report.json',report)
    return report
