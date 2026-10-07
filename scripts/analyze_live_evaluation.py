"""Read live HUD evidence using separately verified glyph calibration."""
import argparse
import json
from pathlib import Path
import sys

import cv2
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from autofarm.realtime.demo_feedback import VerifiedHUDReader, experience_delta, compare_runs
from autofarm.realtime.semantic import atomic_json
from autofarm.realtime.input_audit import assess


def analyze(folder,demonstration,matched_conditions=False,no_manual=False,session_level=True):
    folder=Path(folder);demo=Path(demonstration)
    def metadata(path):
        return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    session=metadata(folder/'session.json')
    configuration=metadata(folder/'configuration_snapshot/run_configuration.json')
    completed=metadata(folder.parent/'report.json')
    human_session=metadata(demo/'session.json')
    verified_live=(session.get('mode')=='LIVE_AUTOMATIC_OBSERVED' and configuration.get('live') is True
        and completed.get('mode')=='LIVE' and completed.get('keys_released') is True
        and session.get('monotonic_origin') is not None
        and session.get('monotonic_origin')!=human_session.get('monotonic_origin'))
    config=json.loads((demo/'analysis/hud_calibration.json').read_text(encoding='utf-8'))
    intervals_path=folder/'intervals.json'
    intervals=json.loads(intervals_path.read_text(encoding='utf-8')) if intervals_path.exists() else {}
    farming_end=intervals.get('farming_ended_at',float('inf'))
    raw=[json.loads(s) for s in (folder/'hud.jsonl').read_text(encoding='utf-8').splitlines()]
    if session_level and raw:
        # The human session's level pixels reject every sample once the character
        # levels up. Anchor the same strict check on this run's own level; a level
        # change inside the run still fails it and stays unknown.
        config=dict(config,level_reference=str((folder/raw[0]['image']).resolve()),level=None)
    reader=VerifiedHUDReader(demo,config)
    readings=[]
    for row in raw:
        if row['t']>farming_end:continue
        row.update(reader.read(cv2.imread(str(folder/row['image']))));readings.append(row)
    # Require both endpoints to agree with a distinct nearby observation. The
    # final recorder duplicate must not count as an independent confirmation.
    confirmed=[]
    for i,row in enumerate(readings):
        if row.get('exp') is None or row.get('level') is None:continue
        peers=readings[max(0,i-3):i]+readings[i+1:i+4]
        if any(0<abs(p['t']-row['t'])<=.7 and p.get('exp')==row['exp'] and p.get('level')==row['level'] for p in peers):
            confirmed.append(row)
    result=dict(net_exp=None,seconds=0,reason='unconfirmed_endpoints')
    if len(confirmed)>=2 and confirmed[0]['t']-readings[0]['t']<=1 and readings[-1]['t']-confirmed[-1]['t']<=1:
        result=experience_delta(confirmed[0],confirmed[-1])
    audit=(assess(folder,confirmed[0]['t'],confirmed[-1]['t']) if verified_live and len(confirmed)>=2
           else dict(confirmed=False,reason='unverified_session_or_interval',manual_interventions=None))
    result.update(mode='LIVE_AUTOMATIC_OBSERVED' if verified_live else 'UNVERIFIED_RECORDING',independent_session=verified_live,
        same_conditions=matched_conditions,manual_interventions=audit['manual_interventions'],input_audit=audit,
        level_reference='scored_session' if session_level and raw else 'human_session',
        first=confirmed[0] if confirmed else None,last=confirmed[-1] if confirmed else None,
        hud_samples=len(readings),unknown_hud_samples=sum(r.get('exp') is None or r.get('level') is None for r in readings),
        effective_damage=None,kills=None,damage_reason='requires deduplicated visual hit evidence')
    result['live_provenance_confirmed']=verified_live
    result['parking_excluded']=bool(intervals)
    human=json.loads((demo/'analysis/human_baseline.json').read_text(encoding='utf-8'))
    human['seconds']=human['hud_seconds']
    result['human_comparison']=compare_runs(human,result)
    result['human_exp_per_minute']=human['exp_per_minute_hud']
    atomic_json(folder/'feedback.json',result)
    with (folder/'hud_readings.jsonl').open('w',encoding='utf-8') as log:
        for r in readings:log.write(json.dumps(r)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder');p.add_argument('demonstration')
    p.add_argument('--matched-conditions',action='store_true');p.add_argument('--no-manual-interventions',action='store_true')
    p.add_argument('--human-level',action='store_true',help='Score against the human session level pixels instead of this run')
    a=p.parse_args();print(json.dumps(analyze(a.folder,a.demonstration,a.matched_conditions,a.no_manual_interventions,
        session_level=not a.human_level),ensure_ascii=False,indent=2))
