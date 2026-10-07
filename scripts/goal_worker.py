"""Fixed local controller entry point; inherits API credentials only in memory."""
import ctypes
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def configure_deepseek_key():
    """User-authorized local fallback; never print or copy credential contents."""
    if os.environ.get('DEEPSEEK_API_KEY'):return
    path=ROOT/'dist'/'deepseek.txt'
    if not path.is_file():return
    value=path.read_text(encoding='utf-8-sig').strip()
    if value.startswith('{'):
        data=json.loads(value)
        value=data.get('DEEPSEEK_API_KEY',data.get('api_key','')) if isinstance(data,dict) else ''
    if not isinstance(value,str) or not 16<=len(value)<=500 or not value.isascii() or any(c.isspace() or ord(c)<32 for c in value):
        raise ValueError('Local DeepSeek credential file has an unsupported format; content omitted')
    os.environ['DEEPSEEK_API_KEY']=value


def execute(job):
    from autofarm.realtime.runtime import run,run_duration_seconds,attack_hold_seconds
    from autofarm.realtime.model import MotionProfile
    from autofarm.winapi import WinApi
    expected={'source_root','folder','seconds','live','online','climb','navigate','record','minimap',
              'attack_hold','policy_mode','policy_model','policy_provider'}
    if not isinstance(job,dict) or set(job)!=expected or Path(job['source_root']).resolve()!=ROOT:
        raise ValueError('Invalid source job')
    folder=Path(job['folder']).resolve()
    if not folder.is_relative_to(ROOT) or folder.parent.name!='runs':raise ValueError('Invalid goal session path')
    for field in ('live','online','climb','navigate','record','minimap'):
        if type(job[field]) is not bool:raise ValueError('Invalid goal switch')
    seconds=run_duration_seconds(job['seconds'])
    if seconds==0 or seconds>600:raise ValueError('Goal trial duration must be bounded to 600 seconds')
    attack_hold=attack_hold_seconds(job['attack_hold'])
    if job.get('policy_provider')=='deepseek' or job.get('policy_provider') is None:
        configure_deepseek_key()
    from autofarm.realtime import runtime,actions,policy
    from autofarm.realtime.semantic import atomic_json
    modules={module.__name__:str(Path(module.__file__).resolve()) for module in (runtime,actions,policy)}
    if any(not Path(path).is_relative_to(ROOT) for path in modules.values()):
        raise RuntimeError('Development controller did not load reviewed source files')
    atomic_json(folder/'backend_source.json',dict(source_root=str(ROOT),modules=modules,
        credentials='inherited process environment; never saved'))
    ctypes.windll.user32.SetProcessDPIAware()
    api=WinApi();hwnd,title=api.find_window('冒险岛怀旧服')
    if title!='冒险岛怀旧服':raise ValueError('Exact game title required')
    adapter=None
    if job['live']:
        from game_input_bridge import GameAdapter
        adapter=GameAdapter();adapter.validate_target(hwnd)
    path=folder/'initial_motion.json'
    motion=MotionProfile.parse(json.loads(path.read_text(encoding='utf-8'))) if path.exists() else None
    parking=job['live'] and (folder/'parking.json').exists()
    return run(api,hwnd,folder,seconds,30,job['live'],adapter,job['navigate'],motion,
        online=job['online'],climb=job['climb'],record=job['record'] and not parking,minimap=job['minimap'],
        evaluation=job['record'] and parking,park_on_finish=parking,hybrid_attacks=True,
        attack_hold=attack_hold,policy_mode=job['policy_mode'],policy_model=job['policy_model'],
        policy_provider=job['policy_provider'])


if __name__=='__main__':
    from autofarm.realtime.semantic import atomic_json
    job=json.loads(Path(sys.argv[2] if sys.argv[1]=='--source-job' else sys.argv[1]).read_text(encoding='utf-8'))
    try:
        report=execute(job)
        print(json.dumps(report,ensure_ascii=False))
    except Exception as error:
        print(type(error).__name__+': '+str(error),file=sys.stderr)
        sys.exit(1)
