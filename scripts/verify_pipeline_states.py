"""Exercise the shared pipeline on existing human observations, never game input."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autofarm.realtime.control import Controller
from autofarm.realtime.model import Actor, Box, Observation, MotionProfile, Platform, Rope
from autofarm.realtime.pipeline import DecisionPipeline
from autofarm.realtime.pipeline_replay import replay_trace


def verify(states, output):
    states, folder = Path(states), Path(output)
    folder.mkdir(parents=True, exist_ok=False)
    controller = Controller(MotionProfile(220,100,150,True), True)
    controller.direct_attacks = True
    pipeline = DecisionPipeline(controller, navigate=True, folder=folder, replay=True)
    count = 0
    first = last = None
    def actor(a):
        return Actor(Box(*a['box']), a['confidence'], a.get('vx',0), a.get('vy',0), a.get('track_id',0)) if a else None
    try:
        with states.open(encoding='utf-8') as rows:
            for line in rows:
                row = json.loads(line)
                t = row['t']
                o = Observation(row['frame_id'], t, actor(row.get('player')),
                    [actor(a) for a in row.get('monsters',[])],
                    [Platform(**p) for p in row['platforms']], [Rope(**r) for r in row['ropes']],
                    reason=row.get('reason',''), navigation_targets=[actor(a) for a in row.get('navigation_targets',[])])
                d = pipeline.step(o,t,offset=tuple(row['camera_offset']),scene_id='recorded-human-observations')
                pipeline.commit(d,False,t,o,simulated=True)
                if first is None:first=t
                last=t;count+=1
    finally:
        pipeline.close(last or 0)
    report = replay_trace(folder)
    report.update(source=str(states.resolve()), source_kind='previously_extracted_human_observations',
        simulated_receipts=True, policy='rule baseline with declared test motion, not a reconstructed live run',
        model_calls=0, real_game_results=False)
    (folder/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('states');p.add_argument('output');a=p.parse_args()
    print(json.dumps(verify(a.states,a.output),ensure_ascii=False,indent=2))
