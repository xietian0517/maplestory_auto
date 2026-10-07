import json
from pathlib import Path
from collections import Counter
r=Path(json.loads(Path('captures/deepseek_goal_20261006/current_trial.json').read_text(encoding='utf-8'))['folder']);s=json.loads((r/'status.json').read_text(encoding='utf-8'));p=s.get('action_policy',{});print(json.dumps(dict(folder=str(r),elapsed=s.get('elapsed_seconds'),phase=s.get('phase'),reason=s.get('reason'),hp=s.get('health',{}).get('hp'),exp=s.get('feedback',{}).get('confirmed_experience'),counts=p.get('counts'),attack_frames=s.get('attack_input_frames'),error=p.get('last_model_error')),ensure_ascii=False))
files=sorted((r/'policy_requests').glob('*/response.json'),key=lambda p:p.stat().st_mtime);print('actions',dict(Counter(json.loads(p.read_text(encoding='utf-8'))['action'] for p in files)))
for f in files[-3:]:
 d=json.loads(f.read_text(encoding='utf-8'));print({k:d.get(k) for k in ('action','target','direction','duration','reason')})
if files:
 d=json.loads((files[-1].parent/'request.json').read_text(encoding='utf-8'))['state'];print('floor',d['player_floor'],'reachable_destinations',len(d['candidate_destinations']))
print('stderr', (r/'backend_stderr.log').read_text(encoding='utf-8')[-200:])

