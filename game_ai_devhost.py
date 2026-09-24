"""Temporary elevated test host; only runs the fixed game controller entry point."""
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
STATE=ROOT/'captures'/'devhost'


def job_args(job,now=None):
    now=time.time() if now is None else now
    if not isinstance(job,dict) or not -1<=now-job['issued_at']<=30:
        raise ValueError('Expired job')
    if job['command'] not in ('play','observe','climb'): raise ValueError('Unknown command')
    folder=(ROOT/job['folder']).resolve()
    if not folder.is_relative_to(ROOT/'captures'): raise ValueError('Session outside captures')
    seconds=job.get('seconds',60)
    if type(seconds) not in (int,float) or not 1<=seconds<=600: raise ValueError('Invalid duration')
    args=[sys.executable,str(ROOT/'game_ai.py'),job['command'],'--folder',str(folder),
          '--seconds',str(seconds),'--delay','2']
    if job.get('navigate'): args.append('--navigate')
    if job.get('record'): args.append('--record')
    if job.get('motion'):
        motion=(ROOT/job['motion']).resolve()
        if not motion.is_relative_to(ROOT/'captures'): raise ValueError('Motion outside captures')
        args+=['--motion',str(motion)]
    return args,folder


def main():
    if not ctypes.windll.shell32.IsUserAnAdmin(): raise RuntimeError('Administrator required')
    from autofarm.realtime.semantic import atomic_json
    from game_input_bridge import GameAdapter
    adapter=GameAdapter()
    STATE.mkdir(parents=True,exist_ok=True)
    job_path=STATE/'job.json'; current=None; folder=None; log=None
    deadline=time.monotonic()+7200
    atomic_json(STATE/'host.json',dict(pid=os.getpid(),phase='ready'))
    try:
        while time.monotonic()<deadline and not (STATE/'STOP').exists():
            if adapter.emergency_pressed(): break
            if current and current.poll() is not None:
                atomic_json(STATE/'host.json',dict(pid=os.getpid(),phase='ready',exit_code=current.returncode))
                log.close(); log=None; current=None
            if not current and job_path.exists():
                try:
                    job=json.loads(job_path.read_text(encoding='utf-8')); job_path.unlink()
                    args,folder=job_args(job)
                    folder.mkdir(parents=True,exist_ok=True)
                    log=(folder/'console.log').open('w',encoding='utf-8')
                    env=dict(os.environ,PYTHONIOENCODING='utf-8')
                    current=subprocess.Popen(args,cwd=ROOT,stdout=log,stderr=log,env=env,
                                             creationflags=subprocess.CREATE_NO_WINDOW)
                    atomic_json(STATE/'host.json',dict(pid=os.getpid(),child_pid=current.pid,phase='testing',folder=str(folder)))
                except Exception as exc:
                    atomic_json(STATE/'host.json',dict(pid=os.getpid(),phase='error',error=str(exc)))
            time.sleep(.05)
    finally:
        if current and current.poll() is None:
            (folder/'STOP').write_text('host stopped',encoding='utf-8')
            try: current.wait(timeout=3)
            except subprocess.TimeoutExpired: current.terminate(); current.wait(timeout=3)
        for key in ('left','right','up','down','alt','shift','home'):
            adapter.send_key(key,True)
        if log: log.close()
        atomic_json(STATE/'host.json',dict(pid=os.getpid(),phase='stopped'))


if __name__=='__main__': main()
