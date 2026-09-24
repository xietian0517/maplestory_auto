"""Experimental map-independent AI controller. Observe mode never sends input."""
import argparse
import ctypes
import json
from pathlib import Path
import sys
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['snapshot','observe','play','climb','plan','stop','replay','host'])
    parser.add_argument('--folder',default='captures/ai_session')
    parser.add_argument('--seconds',type=float,default=60)
    parser.add_argument('--hz',type=float,default=30)
    parser.add_argument('--navigate',action='store_true')
    parser.add_argument('--motion')
    parser.add_argument('--online',action='store_true')
    parser.add_argument('--record',action='store_true',help='Save up to 160 raw frames for offline replay')
    parser.add_argument('--model',default='gpt-6-astra')
    parser.add_argument('--delay',type=float,default=3)
    args=parser.parse_args()
    if args.command=='host':
        from game_ai_devhost import main as host
        host(); return
    folder=Path(args.folder).resolve(); folder.mkdir(parents=True,exist_ok=True)
    if args.command=='replay':
        from autofarm.realtime.replay import replay
        report=replay(folder)
        print(json.dumps({k:v for k,v in report.items() if k!='rows'},ensure_ascii=False,indent=2)); return
    if args.command=='stop':
        (folder/'STOP').write_text('stop',encoding='utf-8'); return
    if args.command=='plan':
        from autofarm.realtime.semantic import OpenAIPlanner
        OpenAIPlanner(args.model).plan(folder); print('GPT scene received and validated'); return
    from autofarm.winapi import WinApi
    from autofarm.realtime.perception import LatestCapture
    from autofarm.realtime.semantic import make_request
    # Prevent DPI virtualization from silently mixing physical and logical pixels.
    ctypes.windll.user32.SetProcessDPIAware()
    api=WinApi(); hwnd,title=api.find_window('冒险岛怀旧服')
    if title!='冒险岛怀旧服': raise ValueError('Exact game title required')
    if args.command=='snapshot':
        with LatestCapture(api,hwnd,args.hz) as capture:
            packet=capture.next(timeout=5)
            if packet is None: raise RuntimeError(capture.error or 'No frame')
            if not packet.foreground: raise RuntimeError('Put game in foreground first')
            data=make_request(packet.image,folder)
            print(json.dumps(dict(folder=str(folder),request_id=data['request_id'],image=data['image'],
                                  width=data['width'],height=data['height']),ensure_ascii=False))
        return
    from autofarm.realtime.runtime import run
    from autofarm.realtime.model import MotionProfile
    adapter=None
    if args.command in ('play','climb'):
        from game_input_bridge import GameAdapter
        adapter=GameAdapter(); adapter.validate_target(hwnd)
    motion=MotionProfile.parse(json.loads(Path(args.motion).read_text(encoding='utf-8'))) if args.motion else None
    if args.navigate and not motion:
        print('Navigation will measure movement and jumping before crossing platforms.',flush=True)
    if (folder/'STOP').exists(): raise RuntimeError('STOP marker exists; choose a fresh session folder')
    time.sleep(max(0,min(10,args.delay)))
    summary=run(api,hwnd,folder,args.seconds,args.hz,args.command in ('play','climb'),adapter,
                args.navigate,motion,args.online,args.model,climb=args.command=='climb',record=args.record)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    try: main()
    except KeyboardInterrupt: print('Stopped')
    except Exception as e:
        # Credentials and API error bodies must never be written to console.
        print(type(e).__name__+': '+str(e),file=sys.stderr); sys.exit(1)
