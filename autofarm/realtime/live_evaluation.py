"""Continuous live evidence, with actual input submissions on the video clock."""
from collections import deque
import hashlib
import json
import sys
from pathlib import Path
import threading
import time

from .demonstration import VideoSegments, write_review
from .semantic import atomic_json


class LiveEvaluation:
    def __init__(self, folder, adapter, hz, clock=time.perf_counter, hwnd=None):
        self.folder=Path(folder)/'evaluation'; self.adapter=adapter; self.hz=hz; self.clock=clock
        self.origin=clock(); self.lock=threading.RLock(); self.held=set()
        self.history=deque([(self.origin,[])],maxlen=8192)
        self.log=None; self.video=None
        self.hwnd=hwnd;self.audit=None

    def mark_farming_end(self):
        atomic_json(self.folder/'intervals.json',dict(farming_ended_at=self.clock()-self.origin,
            following_phase='parking',benchmark_excludes_following_phase=True))

    def __getattr__(self,name):
        return getattr(self.adapter,name)

    def snapshot_files(self,paths,root,destination):
        hashes={}
        for path in paths:
            relative=path.relative_to(root);raw=path.read_bytes()
            target=self.folder/destination/relative;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(raw);hashes[str(relative)]=hashlib.sha256(raw).hexdigest()
        return hashes

    def __enter__(self):
        self.folder.mkdir(parents=True,exist_ok=False)
        self.log=(self.folder/'inputs.jsonl').open('x',encoding='utf-8')
        root=Path(__file__).resolve().parents[2]
        sources=list((root/'autofarm').rglob('*.py'))+[root/'game_ai.py',root/'game_input_bridge.py']
        sources += [p for p in (root/'game_ai_gui.py',root/'scripts'/'goal_worker.py') if p.exists()]
        if getattr(sys,'frozen',False):
            root=Path(sys.executable).parent;sources=[Path(sys.executable)]
        source_hashes=self.snapshot_files(sources,root,'source_snapshot')
        config_root=self.folder.parent
        configs=[p for name in ('request.json','scene.json','world_geometry.json','minimap_calibration.json','parking.json','health.json',
                 'navigation_policy.json','platform_intent.json','farm_plan.json',
                 'run_configuration.json','identity_poses.npy','identity_name.png','monster_templates.npz') if (p:=config_root/name).exists()]
        request=config_root/'request.json'
        if request.exists():
            name=json.loads(request.read_text(encoding='utf-8')).get('image')
            if not isinstance(name,str) or Path(name).name!=name:raise ValueError('Invalid snapshot image name')
            configs.append(config_root/name)
        config_hashes=self.snapshot_files(configs,config_root,'configuration_snapshot')
        atomic_json(self.folder/'session.json',dict(mode='LIVE_AUTOMATIC_OBSERVED',
            monotonic_origin=self.origin,created_at=time.time(),target_fps=self.hz,
            input_timing='successful adapter submissions; not game acknowledgement or human key observations',
            source_sha256=source_hashes,configuration_sha256=config_hashes))
        self.video=VideoSegments(self.folder,self.hz).__enter__()
        if self.hwnd is not None:
            from .input_audit import InputAudit
            self.audit=InputAudit(self.folder,self.hwnd,self.origin,clock=self.clock).start()
        return self

    def send_key(self,key,up=False):
        # Both watchdog releases and main-loop presses pass through this lock.
        with self.lock:
            self.adapter.send_key(key,up)
            now=self.clock()
            if up:self.held.discard(key)
            else:self.held.add(key)
            self.log.write(json.dumps(dict(t=now-self.origin,event='key_up' if up else 'key_down',
                key=key,source='controller_submission'))+'\n');self.log.flush()
            self.history.append((now,sorted(self.held)))

    def offer(self,packet):
        if self.video.error:raise RuntimeError('Evaluation recording failed: '+self.video.error)
        with self.lock:
            sample=next(((t,k) for t,k in reversed(self.history) if t<=packet.started),None)
        row=dict(frame_id=packet.id,capture_started=packet.started-self.origin,
            capture_finished=packet.finished-self.origin,foreground=packet.foreground,
            keys=None if sample is None else sample[1],
            key_sample_age=None if sample is None else packet.started-sample[0],
            key_source='last_successful_controller_submission')
        self.video.offer(packet.image,row)

    def __exit__(self,*args):
        # Runtime puts this context outside LeasedKeys, so releases are recorded.
        if self.audit:self.audit.close()
        if self.video:self.video.__exit__(*args)
        if self.log:
            self.log.write(json.dumps(dict(t=self.clock()-self.origin,event='recording_end',
                held=sorted(self.held),release_observed=not self.held))+'\n');self.log.close()
        atomic_json(self.folder/'report.json',dict(mode='LIVE_AUTOMATIC_OBSERVED',
            elapsed_seconds=self.clock()-self.origin,frames=self.video.written if self.video else 0,
            dropped_frames=self.video.dropped if self.video else 0,error=self.video.error if self.video else None,
            held_at_exit=sorted(self.held),damage=None,kills=None,net_exp=None))
        if self.video and not self.video.error:
            write_review(self.folder)
            path=self.folder/'review.html'
            html=path.read_text(encoding='utf-8').replace('人工示范回看','自动实测回看')
            html=html.replace('按键是采集时刻之前最近一次轮询的状态。','按键是采集时刻之前已成功提交的控制器输入；不代表游戏已执行。')
            path.write_text(html,encoding='utf-8')
