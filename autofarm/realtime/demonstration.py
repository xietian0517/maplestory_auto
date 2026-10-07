"""Read-only human demonstrations: segmented video and timestamped game keys.

No controller or input adapter is constructed. Key edges are observed by polling,
not hardware timestamps; focus boundaries and polling gaps remain explicit.
"""
from collections import Counter, deque
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import queue
import shutil
import sys
import threading
import time

import cv2

from .evidence import InputSessionLock
from .perception import LatestCapture
from .semantic import atomic_json
from .demo_video import H264Writer

VERSION='0.1.1'
GAME_KEYS=('left','right','up','down','shift','alt','ctrl','space',
           'home','end','insert','delete','pgup','pgdn')


class KeyTrace:
    """Pure focus-aware labeling; focus loss is not a human key release."""
    def __init__(self):
        self.focus=None; self.held=set(); self.last=None; self.focus_seconds=0.
        self.max_gap=0.; self.samples=0; self.edges=Counter(); self.history=deque(maxlen=2048)
        self.first=None;self.gap_histogram=Counter()

    def observe(self,t,focus,held):
        held=set(held)&set(GAME_KEYS) if focus else set(); events=[]
        if self.last is not None:
            gap=t-self.last; self.max_gap=max(self.max_gap,gap)
            self.gap_histogram[round(gap*1000000)]+=1
            if self.focus: self.focus_seconds+=gap
            if gap>.03: events.append(dict(t=t,event='sampling_gap',seconds=gap))
        if focus!=self.focus:
            events.append(dict(t=t,event='focus_gained' if focus else 'focus_lost'))
            for key in sorted(self.held): events.append(dict(t=t,event='key_cancel',key=key,reason='focus_lost'))
            for key in sorted(held): events.append(dict(t=t,event='key_sync',key=key,reason='already_held_at_focus_gain'))
        else:
            for key in sorted(self.held-held): events.append(dict(t=t,event='key_up',key=key))
            for key in sorted(held-self.held):
                events.append(dict(t=t,event='key_down',key=key)); self.edges[key]+=1
        if self.first is None:self.first=t
        self.last=t;self.focus=focus;self.held=held;self.samples+=1
        self.history.append((t,focus,sorted(held)))
        return events

    def snapshot(self,t):
        for sampled,focus,held in reversed(self.history):
            if sampled<=t:
                return dict(keys=held,keys_sampled_at=sampled,key_sample_age=t-sampled,key_focus=focus)
        return dict(keys=None,keys_sampled_at=None,key_sample_age=None,key_focus=None)

    def summary(self):
        def percentile(fraction):
            threshold=sum(self.gap_histogram.values())*fraction;count=0
            for micros,n in sorted(self.gap_histogram.items()):
                count+=n
                if count>=threshold:return micros/1000
            return None
        duration=(self.last-self.first) if self.samples>1 else 0
        return dict(foreground_seconds=round(self.focus_seconds,3),key_samples=self.samples,
                    observed_key_poll_hz=round((self.samples-1)/duration,2) if duration else 0,
                    key_poll_gap_p95_ms=percentile(.95),key_poll_gap_p99_ms=percentile(.99),
                    maximum_key_poll_gap_ms=round(self.max_gap*1000,3),key_down_counts=dict(self.edges))


class KeySampler:
    def __init__(self,api,hwnd,folder,origin,stop,hz=240):
        self.api=api;self.hwnd=hwnd;self.folder=Path(folder);self.origin=origin;self.stop=stop;self.hz=hz
        self.trace=KeyTrace();self.lock=threading.Lock();self.done=threading.Event()
        self.error=None;self.stop_reason=None;self.thread=None

    def __enter__(self):
        self.thread=threading.Thread(target=self._run,name='demo-game-keys',daemon=True)
        self.thread.start();return self

    def _run(self):
        try:
            with (self.folder/'inputs.jsonl').open('x',encoding='utf-8') as log:
                flushed=0.
                while not self.done.is_set():
                    started=time.perf_counter()
                    focus=self.api.get_foreground()==self.hwnd and not self.api.is_iconic(self.hwnd)
                    held={key for key in GAME_KEYS if self.api.async_pressed(key)} if focus else set()
                    # Do not associate a key sample with a focus change during polling.
                    focus=focus and self.api.get_foreground()==self.hwnd
                    t=time.perf_counter()-self.origin
                    with self.lock: events=self.trace.observe(t,focus,held)
                    for event in events: log.write(json.dumps(event)+'\n')
                    if focus and self.api.async_pressed('f11'):
                        self.stop_reason='F11';self.stop.set()
                    if t-flushed>=.5 or any(e['event'].startswith('focus_') for e in events):
                        log.flush();flushed=t
                    # Python 3.12 sleep uses a high-resolution waitable timer
                    # on Windows; Event.wait rounded the 4 ms interval to ~15 ms
                    # in the full-resolution recording benchmark. Stop latency
                    # here is bounded by one polling interval.
                    time.sleep(max(0,1/self.hz-(time.perf_counter()-started)))
                log.write(json.dumps(dict(t=time.perf_counter()-self.origin,event='recording_end',
                                          held=sorted(self.trace.held),release_observed=False))+'\n')
        except Exception as exc:
            self.error=type(exc).__name__+': '+str(exc);self.stop.set()

    def snapshot(self,t):
        with self.lock: return self.trace.snapshot(t)

    def summary(self):
        with self.lock: return self.trace.summary()

    def __exit__(self,*args):
        self.done.set();self.thread.join(timeout=3)
        if self.thread.is_alive(): self.error='Key sampler did not finish'


class VideoSegments:
    """Bounded encoding worker. Every written frame has its own capture time."""
    def __init__(self,folder,fps,segment_seconds=30,queue_size=16):
        self.folder=Path(folder);self.fps=fps;self.segment_seconds=segment_seconds
        self.queue=queue.Queue(maxsize=queue_size);self.written=0;self.dropped=0
        self.error=None;self.segments=[];self.thread=None
        self.hud_count=0;self.combat_count=0

    def __enter__(self):
        self.thread=threading.Thread(target=self._run,name='demo-video-writer',daemon=True)
        self.thread.start();return self

    def offer(self,image,row):
        if self.error: return False
        try: self.queue.put_nowait((image.copy(),dict(row)));return True
        except queue.Full: self.dropped+=1;return False

    def _run(self):
        writer=None;segment=None;shape=None;hud_at=-float('inf');combat_at=-float('inf')
        attack_at=-float('inf');last=None;hud_log=None
        def save_hud(image,row,kind):
            hud_folder=self.folder/'hud';hud_folder.mkdir(exist_ok=True)
            name=f'hud/{self.hud_count:06d}.png';y=max(0,image.shape[0]-110)
            if not cv2.imwrite(str(self.folder/name),image[y:]):raise OSError('HUD 截图保存失败')
            sample=dict(image=name,region=[0,y,image.shape[1],image.shape[0]-y],kind=kind,
                        frame_id=row['frame_id'],t=row['capture_started'],segment=row['segment'],video_frame=row['video_frame'])
            hud_log.write(json.dumps(sample)+'\n');hud_log.flush();self.hud_count+=1
            return name,sample['region']
        try:
            with (self.folder/'frames.jsonl').open('x',encoding='utf-8') as log, \
                    (self.folder/'hud.jsonl').open('x',encoding='utf-8') as hud_log, \
                    (self.folder/'combat.jsonl').open('x',encoding='utf-8') as combat_log:
                while True:
                    item=self.queue.get()
                    if item is None: break
                    image,row=item;size=(image.shape[1],image.shape[0]);t=row['capture_started']
                    if writer is None or size!=shape or t-segment['start']>=self.segment_seconds:
                        if writer: writer.release()
                        name=f'segment_{len(self.segments):04d}.mp4'
                        writer=H264Writer(self.folder/name,self.fps,size)
                        if not writer.isOpened(): raise OSError('MP4 编码器无法打开')
                        shape=size;segment=dict(file=name,start=t,end=t,frames=0,width=size[0],height=size[1],fps=self.fps)
                        self.segments.append(segment)
                    writer.write(image)
                    row.update(segment=segment['file'],video_frame=segment['frames'])
                    if t-hud_at>=.2-1e-6:
                        name,region=save_hud(image,row,'initial' if self.hud_count==0 else 'periodic')
                        row.update(hud_image=name,hud_region=region);hud_at=t
                    if 'shift' in (row.get('keys') or []):attack_at=t
                    if t-attack_at<=1.25 and t-combat_at>=.5-1e-6:
                        (self.folder/'combat').mkdir(exist_ok=True)
                        name=f'combat/{self.combat_count:06d}.png'
                        if not cv2.imwrite(str(self.folder/name),image,[cv2.IMWRITE_PNG_COMPRESSION,1]):
                            raise OSError('攻击证据截图保存失败')
                        sample=dict(image=name,t=t,frame_id=row['frame_id'],segment=row['segment'],
                                    video_frame=row['video_frame'],keys=row.get('keys'),reason='attack_input_or_recent_release',
                                    damage_confirmed=False)
                        combat_log.write(json.dumps(sample)+'\n');combat_log.flush()
                        row['combat_image']=name;self.combat_count+=1;combat_at=t
                    log.write(json.dumps(row)+'\n');log.flush()
                    segment['frames']+=1;segment['end']=row['capture_finished'];self.written+=1
                    if segment['frames']==1: atomic_json(self.folder/'segments.json',self.segments)
                    last=(image,row)
                if last is not None:save_hud(*last,'final')
        except Exception as exc: self.error=type(exc).__name__+': '+str(exc)
        finally:
            if writer:
                try:writer.release()
                except Exception as exc:self.error=type(exc).__name__+': '+str(exc)
            atomic_json(self.folder/'segments.json',self.segments)

    def __exit__(self,*args):
        while self.thread.is_alive():
            try: self.queue.put(None,timeout=.1);break
            except queue.Full: continue
        self.thread.join(timeout=30)
        if self.thread.is_alive(): self.error='Video encoder did not finish; last segment may be incomplete'


def provenance():
    if getattr(sys,'frozen',False):
        return dict(executable_sha256=hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest())
    root=Path(__file__).resolve().parents[2]
    names=['autofarm/realtime/demonstration.py','autofarm/realtime/perception.py',
           'autofarm/realtime/evidence.py','autofarm/realtime/demo_video.py',
           'autofarm/realtime/demo_analysis.py','autofarm/winapi.py','game_demo_gui.py']
    return dict(source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest()
                              for name in names if (root/name).exists()})


def write_review(folder):
    """Generate an offline browser player aligned by video-frame index."""
    folder=Path(folder)
    frames=[json.loads(line) for line in (folder/'frames.jsonl').read_text(encoding='utf-8').splitlines()]
    groups={}
    for row in frames: groups.setdefault(row['segment'],[]).append(row)
    data={'segments':json.loads((folder/'segments.json').read_text(encoding='utf-8')),'frames':groups}
    (folder/'review_data.js').write_text('window.DEMO='+json.dumps(data,ensure_ascii=False)+';',encoding='utf-8')
    html='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>人工示范回看</title>
<style>body{background:#111827;color:#e5e7eb;font:17px system-ui;margin:24px}video{display:block;width:min(100%,1366px);margin-top:15px}select,button{font:inherit;padding:8px}#keys{font-size:24px;color:#6ee7b7}p{max-width:1000px}</style>
<h1>人工示范回看</h1><p>按原始帧索引显示采集时间与按键。视频按目标帧率播放；采集不足或失焦时，录像时长可能短于真实用时，以“采集时间”为准。按键是采集时刻之前最近一次轮询的状态。</p>
<select id="parts"></select> <button onclick="v.playbackRate=.25">¼ 速</button> <button onclick="v.playbackRate=1">正常速度</button>
<p id="info"></p><p id="keys"></p><video id="v" controls></video><script src="review_data.js"></script><script>
const v=document.getElementById('v'),parts=document.getElementById('parts'),info=document.getElementById('info'),keys=document.getElementById('keys');
for(const s of DEMO.segments){let o=document.createElement('option');o.value=s.file;o.textContent=s.file+' · '+s.start.toFixed(1)+'–'+s.end.toFixed(1)+' 秒';parts.appendChild(o)}
parts.onchange=()=>{v.src=parts.value};if(DEMO.segments.length)parts.onchange();
function tick(){let s=DEMO.segments.find(s=>s.file===parts.value);if(s){let rows=DEMO.frames[s.file]||[],r=rows[Math.min(rows.length-1,Math.floor(v.currentTime*s.fps))];if(r){info.textContent='采集时间 '+r.capture_started.toFixed(3)+' 秒 · 帧 '+r.frame_id+' · 按键样本间隔 '+(r.key_sample_age==null?'未知':(r.key_sample_age*1000).toFixed(1)+' ms');keys.textContent='按住：'+(r.keys==null?'未知':r.keys.join(' + ')||'无')}}requestAnimationFrame(tick)}tick();</script></html>'''
    (folder/'review.html').write_text(html,encoding='utf-8')


def record_demo(api,hwnd,folder,seconds=600,fps=60,stop=None,notify=lambda value:None,
                capture_factory=LatestCapture,sampler_factory=KeySampler,lock_factory=InputSessionLock):
    if not 1<=seconds<=3600 or fps not in (30,60): raise ValueError('时长应为 1–3600 秒，帧率为 30 或 60')
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    if any((folder/name).exists() for name in ('session.json','inputs.jsonl','frames.jsonl')):
        raise ValueError('请使用新的示范录制目录')
    if shutil.disk_usage(folder).free<4*1024**3: raise OSError('录制盘可用空间不足 4 GB；新版会保留攻击原图和经验栏')
    stop=stop or threading.Event();origin=time.perf_counter();phase='recording';error=None
    received=0;skipped=0;capture_gaps=0;last_id=0;last_status=0;latest_times=deque(maxlen=120)
    sampler=None;video=None;capture=None
    metadata=dict(version=VERSION,mode='HUMAN_DEMONSTRATION_READ_ONLY',created_at=time.time(),
                  monotonic_origin=origin,requested_seconds=seconds,target_fps=fps,key_poll_target_hz=240,
                  keys=list(GAME_KEYS),stop_key='f11',key_timing='observed polling edges, not hardware timestamps',
                  timestamps='seconds since monotonic_origin; video_frame indexes encoded frames',
                  video_encoding='H.264 CRF 18, yuv420p, pad odd dimensions by at most one pixel',
                  hud_sampling='lossless bottom 110 pixels, 5 Hz plus first/final frame; EXP not auto-confirmed',
                  combat_sampling='lossless full frame, 2 Hz during Shift and 1.25s afterwards; damage not auto-confirmed',
                  automatic_input=False,**provenance())
    atomic_json(folder/'session.json',metadata)
    try:
        # Same mutex as every realtime live controller. No input is released,
        # pressed, replayed, or otherwise sent by this recorder.
        with lock_factory(),ExitStack() as stack:
            video=stack.enter_context(VideoSegments(folder,fps))
            sampler=stack.enter_context(sampler_factory(api,hwnd,folder,origin,stop))
            capture=stack.enter_context(capture_factory(api,hwnd,fps,foreground_only=True))
            journal=stack.enter_context((folder/'capture.jsonl').open('x',encoding='utf-8'))
            while not stop.is_set() and time.perf_counter()-origin<seconds:
                packet=capture.next(last_id,timeout=.15);now=time.perf_counter();elapsed=now-origin
                if capture.error or sampler.error or video.error:
                    raise RuntimeError(capture.error or sampler.error or video.error)
                if packet is not None:
                    capture_gaps+=max(0,packet.id-last_id-1);last_id=packet.id
                    row=dict(frame_id=packet.id,capture_started=packet.started-origin,
                             capture_finished=packet.finished-origin,region=packet.region,foreground=packet.foreground)
                    if packet.foreground and api.get_foreground()==hwnd:
                        row.update(sampler.snapshot(packet.finished-origin))
                        row['queued']=video.offer(packet.image,row);received+=1;latest_times.append(elapsed)
                        if received==1: cv2.imwrite(str(folder/'first_frame.png'),packet.image)
                    else: row.update(queued=False,skip_reason='focus_lost');skipped+=1
                    journal.write(json.dumps(row)+'\n')
                if elapsed-last_status>=.5:
                    journal.flush();stats=sampler.summary()
                    active=stats['foreground_seconds']
                    actual=(len(latest_times)-1)/(latest_times[-1]-latest_times[0]) if len(latest_times)>1 and elapsed-latest_times[-1]<1 else 0
                    status=dict(phase='recording',elapsed_seconds=round(elapsed,2),requested_seconds=seconds,
                                target_fps=fps,recent_capture_fps=round(actual,2),frames_received=received,
                                frames_written=video.written,encoder_drops=video.dropped,capture_mailbox_gaps=capture_gaps,
                                hud_samples=video.hud_count,combat_samples=video.combat_count,
                                backend=capture.backend,foreground=api.get_foreground()==hwnd,**stats)
                    atomic_json(folder/'status.json',status);notify(status);last_status=elapsed
                    if shutil.disk_usage(folder).free<512*1024**2: raise OSError('剩余空间不足 512 MB，已停止录制')
            phase=sampler.stop_reason or ('stopped' if stop.is_set() else 'completed')
            notify(dict(phase='saving',elapsed_seconds=time.perf_counter()-origin))
        if video.error or sampler.error: raise RuntimeError(video.error or sampler.error)
        if video.written==0: raise RuntimeError('没有录到游戏前台画面，请检查窗口是否可见')
        write_review(folder)
    except Exception as exc:
        phase='error';error=type(exc).__name__+': '+str(exc)
        if 'Another AI controller' in error:
            error='旧挂机程序仍在运行。请先按 F11 或点击旧程序的停止按钮，再开始示范录制。'
    finally:
        elapsed=time.perf_counter()-origin;stats=sampler.summary() if sampler else {}
        active=stats.get('foreground_seconds',0)
        report=dict(phase=phase,error=error,automatic_input=False,elapsed_seconds=round(elapsed,3),
                    frames_received=received,frames_written=video.written if video else 0,
                    hud_samples=video.hud_count if video else 0,combat_samples=video.combat_count if video else 0,
                    encoder_drops=video.dropped if video else 0,capture_mailbox_gaps=capture_gaps,
                    focus_boundary_skipped_frames=skipped,backend=capture.backend if capture else None,
                    average_capture_fps_foreground=round(received/active,2) if active else 0,
                    target_fps=fps,**stats)
        atomic_json(folder/'report.json',report);atomic_json(folder/'status.json',report)
    if (folder/'inputs.jsonl').exists() and (folder/'frames.jsonl').exists():
        try:
            from .demo_analysis import analyze
            result=analyze(folder,extract=False)
            report['analysis']='analysis/analysis.json';report['quality_issues']=result['quality_issues']
        except Exception as exc:report['analysis_error']=type(exc).__name__+': '+str(exc)
        atomic_json(folder/'report.json',report);atomic_json(folder/'status.json',report)
    notify(report)
    return report
