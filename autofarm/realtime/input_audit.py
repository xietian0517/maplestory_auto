"""Read-only Windows input provenance. Never blocks, modifies or sends input.

Only game-foreground external events are retained. Non-control keys are
redacted to `other`; no typed text, mouse coordinates or other-app input.
Microsoft documents silent hook removal on timeout, so installation alone is
not coverage proof. The offline audit also matches every controller edge.
https://learn.microsoft.com/en-us/windows/win32/winmsg/lowlevelkeyboardproc
https://learn.microsoft.com/en-us/windows/win32/api/winuser/ns-winuser-kbdllhookstruct
"""
from collections import deque
import ctypes as C
from ctypes import wintypes as W
import json
import os
from pathlib import Path
import threading
import time

from autofarm.winapi import INPUT_SOURCE_TAG
from .semantic import atomic_json


CONTROL_KEYS={0x25:'left',0x26:'up',0x27:'right',0x28:'down',0x24:'home',
              0x10:'shift',0xA0:'shift',0x12:'alt',0xA4:'alt',0x1B:'esc',0x7A:'f11'}


def keyboard_event(vk,flags,extra,foreground,t):
    owned=bool(flags&0x10) and extra==INPUT_SOURCE_TAG
    if not foreground and not owned:return None
    return dict(t=t,device='keyboard',key=CONTROL_KEYS.get(vk,'other'),
                event='key_up' if flags&0x80 else 'key_down',foreground=bool(foreground),
                source='controller' if owned else 'external_injected' if flags&0x10 else 'physical')


class InputAudit:
    def __init__(self,folder,hwnd,origin,clock=time.perf_counter):
        self.folder=Path(folder);self.hwnd=hwnd;self.origin=origin;self.clock=clock
        self.ready=threading.Event();self.stop=threading.Event();self.thread=None
        self.queue=deque();self.errors=[];self.installed=False;self.started=None;self.ended=None
        self.max_pump_gap=0.;self.max_callback=0.;self.events=0

    def fail(self,reason):
        if reason not in self.errors:self.errors.append(reason)

    def start(self):
        if os.name!='nt':self.fail('unsupported_platform');return self
        self.thread=threading.Thread(target=self._run,name='game-input-audit',daemon=True)
        self.thread.start()
        if not self.ready.wait(2):self.fail('startup_timeout')
        return self

    def append(self,row):
        if row is None:return
        if len(self.queue)>=10000:self.fail('event_queue_overflow')
        else:self.queue.append(row)

    def _run(self):
        hooks=[];u=None
        try:
            class Keyboard(C.Structure):
                _fields_=[('vk',W.DWORD),('scan',W.DWORD),('flags',W.DWORD),
                          ('time',W.DWORD),('extra',C.c_size_t)]
            class Mouse(C.Structure):
                _fields_=[('point',W.POINT),('data',W.DWORD),('flags',W.DWORD),
                          ('time',W.DWORD),('extra',C.c_size_t)]
            u=C.WinDLL('user32',use_last_error=True);k=C.WinDLL('kernel32',use_last_error=True)
            proc=C.WINFUNCTYPE(C.c_ssize_t,C.c_int,W.WPARAM,W.LPARAM)
            u.SetWindowsHookExW.argtypes=[C.c_int,proc,W.HINSTANCE,W.DWORD]
            u.SetWindowsHookExW.restype=W.HANDLE
            u.CallNextHookEx.argtypes=[W.HANDLE,C.c_int,W.WPARAM,W.LPARAM]
            u.CallNextHookEx.restype=C.c_ssize_t
            u.UnhookWindowsHookEx.argtypes=[W.HANDLE];u.UnhookWindowsHookEx.restype=W.BOOL
            u.GetForegroundWindow.restype=W.HWND
            u.PeekMessageW.argtypes=[C.POINTER(W.MSG),W.HWND,W.UINT,W.UINT,W.UINT]
            u.PeekMessageW.restype=W.BOOL
            u.DispatchMessageW.argtypes=[C.POINTER(W.MSG)];u.DispatchMessageW.restype=C.c_ssize_t
            k.GetModuleHandleW.argtypes=[W.LPCWSTR];k.GetModuleHandleW.restype=W.HMODULE

            def receive(device,code,wp,lp):
                began=self.clock()
                try:
                    if code==0:
                        foreground=u.GetForegroundWindow()==self.hwnd
                        if device=='keyboard':
                            e=C.cast(lp,C.POINTER(Keyboard)).contents
                            self.append(keyboard_event(e.vk,e.flags,e.extra,foreground,began-self.origin))
                        elif foreground:
                            e=C.cast(lp,C.POINTER(Mouse)).contents
                            self.append(dict(t=began-self.origin,device='mouse',
                                event='move' if wp==0x200 else 'button_or_wheel',foreground=True,
                                source='external_injected' if e.flags&1 else 'physical'))
                except Exception as exc:self.fail('callback_'+type(exc).__name__)
                finally:
                    duration=self.clock()-began;self.max_callback=max(self.max_callback,duration)
                    if duration>.1:self.fail('slow_callback')
                return u.CallNextHookEx(None,code,wp,lp)

            keyboard=proc(lambda n,w,l:receive('keyboard',n,w,l))
            mouse=proc(lambda n,w,l:receive('mouse',n,w,l))
            # Keep both ctypes callback objects alive until both hooks detach.
            for kind,callback in ((13,keyboard),(14,mouse)):
                handle=u.SetWindowsHookExW(kind,callback,k.GetModuleHandleW(None),0)
                if not handle:raise OSError(C.get_last_error(),'Input audit hook installation failed')
                hooks.append(handle)
            self.installed=True;self.started=self.clock()-self.origin;self.ready.set()
            with (self.folder/'input_audit.jsonl').open('x',encoding='utf-8') as log:
                message=W.MSG();last=self.clock();heartbeat=last
                while not self.stop.is_set():
                    while u.PeekMessageW(C.byref(message),None,0,0,1):
                        u.DispatchMessageW(C.byref(message))
                    now=self.clock();gap=now-last;last=now
                    self.max_pump_gap=max(self.max_pump_gap,gap)
                    if gap>.25:self.fail('message_pump_gap')
                    while self.queue:
                        log.write(json.dumps(self.queue.popleft())+'\n');self.events+=1
                    if now-heartbeat>=1:
                        log.write(json.dumps(dict(t=now-self.origin,event='heartbeat',
                            foreground=u.GetForegroundWindow()==self.hwnd))+'\n');log.flush();heartbeat=now
                    self.stop.wait(.005)
                while self.queue:
                    log.write(json.dumps(self.queue.popleft())+'\n');self.events+=1
        except Exception as exc:self.fail(type(exc).__name__+': '+str(exc))
        finally:
            if u:
                for handle in hooks:
                    if not u.UnhookWindowsHookEx(handle):self.fail('unhook_failed')
            self.ended=self.clock()-self.origin;self.ready.set()

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join(2)
            if self.thread.is_alive():self.fail('shutdown_timeout')
        result=dict(mode='WINDOWS_GAME_INPUT_AUDIT',installed=self.installed,start=self.started,end=self.ended,
                    errors=self.errors,event_count=self.events,max_pump_gap=self.max_pump_gap,
                    max_callback_seconds=self.max_callback,controller_tag=INPUT_SOURCE_TAG,
                    scope='Game-foreground keyboard/buttons/wheel; controller-tagged edges include releases after focus loss. Other keys redacted; mouse motion counted separately.',
                    limitation='Hook installation alone cannot prove coverage; compare tagged events with every submitted input.')
        atomic_json(self.folder/'input_audit_report.json',result)
        return result


def assess(folder,start,end):
    """Bound the claim to a measured interval; never replace missing data with 0."""
    folder=Path(folder)
    report=folder/'input_audit_report.json';events=folder/'input_audit.jsonl'
    if not report.exists() or not events.exists():return dict(confirmed=False,reason='audit_missing',manual_interventions=None)
    info=json.loads(report.read_text(encoding='utf-8'))
    if (info.get('mode')!='WINDOWS_GAME_INPUT_AUDIT' or not info.get('installed') or info.get('errors')
            or info.get('start') is None or info['start']>start or info.get('end') is None or info['end']<end):
        return dict(confirmed=False,reason='audit_coverage_unknown',manual_interventions=None)
    rows=[json.loads(s) for s in events.read_text(encoding='utf-8').splitlines()]
    own=[r for r in rows if r.get('source')=='controller']
    submitted=[json.loads(s) for s in (folder/'inputs.jsonl').read_text(encoding='utf-8').splitlines()]
    submitted=[r for r in submitted if r.get('event') in ('key_down','key_up')]
    used=set();missing=0;interval_count=0;matched_count=0
    for s in submitted:
        inside=start<=s['t']<=end
        interval_count+=inside
        matches=[(abs(r['t']-s['t']),i) for i,r in enumerate(own) if i not in used
                 and r.get('event')==s['event'] and r.get('key')==s['key'] and abs(r['t']-s['t'])<=.15]
        if not matches:missing+=inside
        else:
            used.add(min(matches)[1]);matched_count+=inside
    extra=sum(start<=r['t']<=end and i not in used for i,r in enumerate(own))
    external=[r for r in rows if start<=r['t']<=end and r.get('source') in ('physical','external_injected')
              and r.get('foreground') and r.get('event')!='move']
    motion=sum(start<=r['t']<=end and r.get('event')=='move' for r in rows)
    confirmed=bool(interval_count) and missing==0 and extra==0
    return dict(confirmed=confirmed,reason='all_submissions_observed' if confirmed else 'submitted_edges_not_fully_matched',
                submitted_edges=interval_count,matched_edges=matched_count,missing_edges=missing,extra_tagged_edges=extra,
                external_events=len(external),mouse_motion_events=motion,
                manual_interventions=len(external) if confirmed else None)
