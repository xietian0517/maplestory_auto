"""Require a recorded ten-minute active AI session before reporting goal success."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.score_safe_ledge_run import score
from autofarm.realtime.semantic import atomic_json


def assess_ai(run,comparison):
    run=Path(run)
    config=json.loads((run/'action_policy_configuration.json').read_text(encoding='utf-8'))
    run_config=json.loads((run/'run_configuration.json').read_text(encoding='utf-8'))
    report=json.loads((run/'report.json').read_text(encoding='utf-8'))
    farming=report.get('farming_summary') or report
    policy=farming.get('action_policy') or report.get('action_policy') or {}
    counts=policy.get('counts') or {}
    effective_ai=(config.get('mode')=='active' and config.get('fallback')=='wait'
        and counts.get('responses',0)>0 and counts.get('accepted',0)>0
        and counts.get('source_ai_executor',0)>0 and counts.get('source_fallback_rule',0)==0
        and farming.get('attack_input_frames',0)>0)
    complete=(run_config.get('seconds')==600 and farming.get('elapsed_seconds',0)>=599
        and comparison.get('complete_verified_run') is True)
    requests=list((run/'policy_requests').glob('*/response.json'))
    intents=[json.loads(p.read_text(encoding='utf-8')) for p in requests]
    elapsed=farming.get('elapsed_seconds',0)
    result=dict(version=1,run=str(run.resolve()),requested_seconds=run_config.get('seconds'),
        real_ten_minute_verified=complete,active_ai_verified=effective_ai,
        target_met=bool(complete and effective_ai and comparison.get('target_met')),
        measured=comparison.get('automatic'),target_exp_per_minute=comparison.get('target_exp_per_minute'),
        reference_ratio=comparison.get('ratio'),model=config.get('model'),provider=config.get('provider'),
        counts=counts,archived_responses=len(intents),
        chosen_lane_responses=sum(i.get('target') in ('lane:left','lane:right') for i in intents),
        waiting_cycles=counts.get('source_fallback_wait',0),
        ai_cycles=counts.get('source_ai_executor',0),
        active_input_frames=farming.get('active_input_frames',0),
        attack_input_frames=farming.get('attack_input_frames',0),
        elapsed_seconds=elapsed,input_audit=comparison.get('input_audit'),
        keys_released=report.get('keys_released'),parking=report.get('parking'),
        limitation='One live run; historical human reference may differ in level, buffs, spawns and other players.')
    atomic_json(run/'ai_goal_score.json',result)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    parser.add_argument('--human',type=Path,default=ROOT/'dist/demonstrations/demo_20260927_130349_297600')
    parser.add_argument('--alphabet',type=Path,default=ROOT/'dist/demonstrations/demo_20260925_130158_91500')
    args=parser.parse_args()
    comparison=score(args.run,args.human,args.alphabet)
    result=assess_ai(args.run,comparison)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
