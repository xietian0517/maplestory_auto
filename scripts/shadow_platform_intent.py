"""Check learned platform suggestions against a completed live trace, without inputs."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from autofarm.realtime.control import graph,route,standing_platform
from autofarm.realtime.demo_policy import PlatformIntent
from autofarm.realtime.model import Actor,Box,MotionProfile,Observation,Platform,Rope
from autofarm.realtime.semantic import atomic_json


def shadow(folder,model_file,out):
    folder=Path(folder);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    model=PlatformIntent(json.loads(Path(model_file).read_text()))
    world=json.loads((folder/'evaluation/configuration_snapshot/world_geometry.json').read_text())
    config=json.loads((folder/'evaluation/configuration_snapshot/run_configuration.json').read_text())
    motion=MotionProfile.parse(config['initial_motion']);counts=Counter();suggestions=Counter();examples=[];next_at=0
    invalid={'player_not_found','identity_confirming','camera_or_map_changed','stale_frame','focus_lost','window_resized'}
    nav={'search','patrol','approach_launch','rope_approach','edge_recovery','route_retry_pending'}
    started=time.perf_counter()
    with (out/'predictions.jsonl').open('x',encoding='utf-8') as log:
        for line in (folder/'frames.jsonl').open(encoding='utf-8'):
            r=json.loads(line)
            if r.get('phase')!='farming' or r['t']<next_at:continue
            next_at=r['t']+.19
            if not r.get('player') or r['reason'] in invalid:counts['identity_or_frame_unknown']+=1;continue
            x,y=r['player'];dx,dy=r['camera_offset'];vx,vy=r['player_velocity']
            ps=[Platform(p['id'],p['left']+dx,p['right']+dx,p['y']+dy) for p in world['platforms']]
            ropes=[Rope(p['x']+dx,p['top']+dy,p['bottom']+dy) for p in world['ropes']]
            o=Observation(r['frame_id'],r['t'],Actor(Box(x-15,y-48,x+15,y),1,vx,vy),platforms=ps,ropes=ropes)
            floor=standing_platform(o)
            if floor is None or abs(vy)>=60:counts['not_grounded']+=1;continue
            def actors(boxes):return [dict(box=b,confidence=.78) for b in boxes]
            record=dict(player=dict(box=list(vars(o.player.box).values()),vx=vx,vy=vy),
                floor_id=floor.id,reason='',platforms=[vars(p) for p in ps],
                monsters=actors(r.get('monsters',[])),navigation_targets=actors(r.get('navigation_targets',[])))
            ranking=model.predict(record);goal,score=ranking[0]
            reachable=goal==floor.id or bool(route(graph(ps,ropes,motion),floor.id,goal))
            accepted=score>=.55 and goal!=floor.id and reachable
            counts['grounded_predictions']+=1;counts['top_same_floor']+=goal==floor.id
            counts['top_unreachable']+=not reachable;counts['high_vote_reachable_change']+=accepted
            if accepted:suggestions[goal]+=1
            result=dict(t=r['t'],frame_id=r['frame_id'],current_floor=floor.id,actual_reason=r['reason'],
                        top_three=ranking[:3],reachable=reachable,eligible_suggestion=accepted,
                        route_selection_frame=r['reason'] in nav)
            log.write(json.dumps(result)+'\n')
            if accepted and r['reason'] in nav and len(examples)<12:examples.append(result)
    report=dict(mode='OFFLINE_SHADOW_ON_RECORDED_LIVE_STATE',counts=dict(counts),suggestions=dict(suggestions),
        examples=examples,wall_seconds=time.perf_counter()-started,automatic_inputs=False,live_exp_per_minute=None,
        caveats=['Uses saved initial motion and fixed observed atlas; dynamic support and refined speed are not reconstructed.',
                 'Trace stores monster boxes without confidence; accepted boxes are conservatively represented at 0.78.',
                 'A reachable suggestion is not evidence of a successful action or improved income.'])
    atomic_json(out/'report.json',report);return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('model');p.add_argument('output');a=p.parse_args()
    r=shadow(a.run,a.model,a.output);print(json.dumps({k:v for k,v in r.items() if k!='examples'},indent=2))
