"""Bounded background recording; disk compression never blocks the input loop."""
import json
import hashlib
from pathlib import Path
import queue
import threading
from contextlib import nullcontext

import cv2


class EvidenceRecorder:
    def __init__(self,folder,limit=160,hz=10):
        self.folder=Path(folder)/'replay'; self.limit=limit; self.interval=1/hz
        self.queue=queue.Queue(maxsize=4); self.count=0; self.last=-float('inf')
        self.dropped=0; self.error=None; self.thread=None
        self.bound_scenes=set()

    def bind_scene(self,scene,seed):
        if scene.request_id in self.bound_scenes: return
        # Scene changes are rare. Preserve the exact seed/annotations associated
        # with each frame so later replanning cannot invalidate old recordings.
        self.queue.put(('scene',(scene.to_data(),seed.copy())),timeout=2)
        self.bound_scenes.add(scene.request_id)

    def preview(self,image):
        try: self.queue.put_nowait(('preview',image.copy()))
        except queue.Full: pass

    def __enter__(self):
        (self.folder if self.limit else self.folder.parent).mkdir(parents=True,exist_ok=True)
        if self.limit and (self.folder/'frames.jsonl').exists():
            raise ValueError('Replay already exists; choose a new session folder')
        self.thread=threading.Thread(target=self._write,daemon=True); self.thread.start()
        return self

    def offer(self,packet,row):
        if self.error or self.count>=self.limit or packet.started-self.last<self.interval: return
        item=dict(row,image=f'{packet.id:06d}.png',captured_at=packet.started)
        try: self.queue.put_nowait(('frame',(packet.image.copy(),item)))
        except queue.Full: self.dropped+=1; return
        self.count+=1; self.last=packet.started

    def _write(self):
        try:
            with ((self.folder/'frames.jsonl').open('w',encoding='utf-8') if self.limit else nullcontext(None)) as log:
                while True:
                    item=self.queue.get()
                    if item is None: break
                    kind,payload=item
                    if kind=='preview':
                        cv2.imwrite(str(self.folder.parent/'latest.png'),payload)
                        continue
                    if kind=='scene':
                        from .semantic import atomic_json
                        scene,seed=payload; dest=self.folder/'scenes'/scene['request_id']; dest.mkdir(parents=True,exist_ok=True)
                        ok,png=cv2.imencode('.png',seed)
                        if not ok: raise OSError('Could not encode replay seed')
                        raw=png.tobytes(); (dest/'seed.png').write_bytes(raw)
                        atomic_json(dest/'scene.json',scene)
                        atomic_json(dest/'request.json',dict(request_id=scene['request_id'],width=scene['width'],height=scene['height'],
                            image='seed.png',image_sha256=hashlib.sha256(raw).hexdigest()))
                        continue
                    image,row=payload
                    if not cv2.imwrite(str(self.folder/row['image']),image,[cv2.IMWRITE_PNG_COMPRESSION,1]):
                        raise OSError('Could not write replay frame')
                    log.write(json.dumps(row)+'\n'); log.flush()
        except Exception as exc: self.error=type(exc).__name__+': '+str(exc)

    def __exit__(self,*args):
        while self.thread.is_alive():
            try: self.queue.put(None,timeout=.1); break
            except queue.Full: continue
        self.thread.join(timeout=5)


class InputSessionLock:
    """Single active controller per Windows desktop session, across CLI and GUI."""
    def __enter__(self):
        import ctypes as C
        from ctypes import wintypes as W
        self.api=C.WinDLL('kernel32',use_last_error=True)
        self.api.CreateMutexW.argtypes=[C.c_void_p,W.BOOL,W.LPCWSTR]
        self.api.CreateMutexW.restype=W.HANDLE
        self.api.CloseHandle.argtypes=[W.HANDLE]
        self.handle=self.api.CreateMutexW(None,False,r'Local\MapleAIControllerInput-v1')
        if not self.handle: raise RuntimeError('Could not acquire game controller lock')
        if C.get_last_error()==183:
            self.api.CloseHandle(self.handle); self.handle=None
            raise RuntimeError('Another AI controller is already running')
        return self

    def __exit__(self,*args):
        if self.handle: self.api.CloseHandle(self.handle); self.handle=None
