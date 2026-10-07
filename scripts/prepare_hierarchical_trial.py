"""Prepare a reproducible goal session; starting input is a separate explicit step."""
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from game_ai_gui import prepare_session
from autofarm.realtime.policy import PolicyConfig
from autofarm.realtime.semantic import atomic_json


def main():
    source=ROOT/'releases/v0.5.6/captures/monkey_forest_v046'
    folder=ROOT/'captures/runs'/('hierarchical_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    prepare_session(source,folder)
    config=PolicyConfig(mode='active',provider='deepseek',model='deepseek-flash',
        timeout=30,response_ttl=30,allow_transfers=True,allow_map_transit=True,
        deepseek_thinking=True,max_output_tokens=8192,hierarchical=True)
    atomic_json(folder/'ai_policy.json',config.data())
    job=dict(source_root=str(ROOT),folder=str(folder),seconds=600,live=True,online=False,
             climb=False,navigate=True,record=True,minimap=True,attack_hold=.3,
             policy_mode='active',policy_model='deepseek-flash',policy_provider='deepseek')
    job_path=folder/'goal_job.json';atomic_json(job_path,job)
    atomic_json(ROOT/'captures/deepseek_goal_20261006/current_trial.json',dict(folder=str(folder),job=str(job_path)))
    print(json.dumps(dict(folder=str(folder),job=str(job_path))))


if __name__=='__main__':main()
