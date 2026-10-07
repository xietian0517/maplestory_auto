"""Prepare a fresh live comparison session using the exact v0.4.4 preset."""
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'variants/v044_ai'
sys.path.insert(0,str(SOURCE))
from game_ai_gui import prepare_session
from autofarm.realtime.semantic import atomic_json


def main():
    release=ROOT/'releases/v0.4.4-ai1'
    folder=SOURCE/'captures/runs'/('v044_ai_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    prepare_session(release/'captures/monkey_forest_v044',folder)
    atomic_json(folder/'ai_strategy.json',dict(version=1,enabled=True,credential_file=str(ROOT/'dist/deepseek.txt')))
    pointer=dict(baseline='0.4.4',source_root=str(SOURCE),folder=str(folder),seconds=600)
    atomic_json(ROOT/'captures/deepseek_goal_20261006/current_v044_trial.json',pointer)
    print(json.dumps(pointer))


if __name__=='__main__':main()
