"""Score an observed live run against the latest reviewed human demonstration."""
import argparse
import json
from pathlib import Path
import sys

import cv2

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from autofarm.realtime.demo_feedback import VerifiedHUDReader,experience_delta
from autofarm.realtime.input_audit import assess


def readings(folder,alphabet):
    config=json.loads((alphabet/'analysis/hud_calibration.json').read_text(encoding='utf-8'))
    rows=[json.loads(s) for s in (folder/'hud.jsonl').read_text(encoding='utf-8').splitlines()]
    config.update(level_reference=str((folder/rows[0]['image']).resolve()),level=None)
    reader=VerifiedHUDReader(alphabet,config)
    result=[dict(t=r['t'],image=r['image'],**reader.read(cv2.imread(str(folder/r['image'])))) for r in rows]
    end_path=folder/'intervals.json'
    if end_path.exists():
        end=json.loads(end_path.read_text(encoding='utf-8'))['farming_ended_at']
        result=[r for r in result if r['t']<=end]
    valid=[r for i,r in enumerate(result) if r['exp'] is not None and r['level'] is not None
        and any(0<abs(r['t']-n['t'])<=.7 and r['exp']==n['exp'] and r['level']==n['level']
            for n in result[max(0,i-3):i]+result[i+1:i+4])]
    if len(valid)<2 or valid[0]['t']-result[0]['t']>1 or result[-1]['t']-valid[-1]['t']>1:
        raise ValueError('Unconfirmed scoring endpoints')
    score=experience_delta(valid[0],valid[-1])
    score.update(first=valid[0],last=valid[-1],samples=len(result),
                 unknown=sum(r['exp'] is None or r['level'] is None for r in result))
    return score


def score(run,human,alphabet):
    baseline=readings(human,alphabet);live=readings(run/'evaluation',alphabet)
    config=json.loads((run/'evaluation/configuration_snapshot/run_configuration.json').read_text(encoding='utf-8'))
    session=json.loads((run/'evaluation/session.json').read_text(encoding='utf-8'))
    report_path=run/'report.json'
    report=json.loads(report_path.read_text(encoding='utf-8')) if report_path.exists() else {}
    audit=assess(run/'evaluation',live['first']['t'],live['last']['t'])
    ratio=live['exp_per_minute']/baseline['exp_per_minute']
    valid=(session['mode']=='LIVE_AUTOMATIC_OBSERVED' and config['live'] is True
        and report.get('keys_released') is True and live['seconds']>=590 and not live['unknown']
        and audit.get('confirmed') is True and audit.get('manual_interventions')==0)
    result=dict(human=str(human.resolve()),baseline=baseline,automatic=live,target_ratio=.8,
        target_exp_per_minute=baseline['exp_per_minute']*.8,ratio=ratio,
        complete_verified_run=valid,target_met=bool(valid and ratio>=.8),input_audit=audit,
        parking=report.get('parking'),limitation='Single run; monster spawns and other players may differ.')
    (run/'score.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path)
    p.add_argument('--human',type=Path,default=Path('dist/demonstrations/demo_20260927_130349_297600'))
    p.add_argument('--alphabet',type=Path,default=Path('dist/demonstrations/demo_20260925_130158_91500'))
    args=p.parse_args();print(json.dumps(score(args.run,args.human,args.alphabet),ensure_ascii=False,indent=2))
