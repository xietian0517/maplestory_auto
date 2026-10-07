import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from autofarm.realtime.climbing import RopeClimber
from autofarm.realtime.control import Controller, graph
from autofarm.realtime.model import Actor, Box, Decision, MotionProfile, Observation, Platform, Rope
from autofarm.realtime.runtime import Metrics


def observation(t, y=400, x=150, vy=0):
    return Observation(1,t,Actor(Box(x-15,y-48,x+15,y),.99,vy=vy),
        platforms=[Platform('source',0,300,400),Platform('target',0,300,150),
                   Platform('lower',0,300,500)],ropes=[Rope(150,150,360)],map_epoch=1)


class RopeFailureTests(unittest.TestCase):
    def test_recorded_late_attachment_does_not_release_up_at_old_timeout(self):
        data=json.loads((Path(__file__).parent/'fixtures/realtime/late_rope_attachment.json').read_text())
        c=RopeClimber('target',jump_height=data['jump_height'])
        decisions=[]
        for r in data['rows']:
            if r['player'] is None:continue
            x,y=r['player'];dx,dy=r['camera_offset'];vx,vy=r['player_velocity']
            platforms=[Platform(p['id'],p['left']+dx,p['right']+dx,p['y']+dy) for p in data['platforms']]
            rope=data['rope']
            o=Observation(r['frame_id'],r['t'],Actor(Box(x-15,y-48,x+15,y),.99,vx=vx,vy=vy),
                platforms=platforms,ropes=[Rope(rope['x']+dx,rope['top']+dy,rope['bottom']+dy)])
            decisions.append(c.decide(o,r['t']))
        self.assertEqual(data['rows'][-1]['reason'],'rope_retry')
        self.assertEqual(decisions[-1].reason,'rope_probe_attachment')
        self.assertEqual(decisions[-1].keys,{'up'})
        self.assertFalse(c.grab_confirmed) # No invented outcome after replay diverges.

    def test_late_attachment_keeps_up_and_requires_observed_rise(self):
        c=RopeClimber('target',jump_height=113)
        c.decide(observation(1),1)
        d=c.decide(observation(2.26,y=308,vy=-90),2.26)
        self.assertEqual(d.reason,'rope_probe_attachment')
        self.assertEqual(d.keys,{'up'});self.assertFalse(c.grab_confirmed)
        d=c.decide(observation(2.42,y=292,vy=-90),2.42)
        self.assertEqual(d.reason,'rope_ascend');self.assertTrue(c.grab_confirmed)

    def test_probe_falling_or_stalled_is_not_success(self):
        for y in (308,330):
            c=RopeClimber('target',jump_height=113);c.decide(observation(1),1)
            c.decide(observation(2.26,y=308),2.26)
            d=c.decide(observation(2.92,y=y),2.92)
            self.assertEqual(d.reason,'rope_resume_unconfirmed')
            self.assertFalse(d.keys);self.assertFalse(c.grab_confirmed)

    def test_other_landing_ends_transfer_without_another_jump(self):
        c=RopeClimber('target');c.decide(observation(1),1)
        self.assertFalse(c.decide(observation(2,y=500),2).keys)
        d=c.decide(observation(2.13,y=500),2.13)
        self.assertEqual(d.reason,'rope_landed_elsewhere');self.assertEqual(c.phase,'failed')

    def test_interrupted_transfer_keeps_failure_budget_across_reset(self):
        c=Controller(navigate=True);c.last_epoch=1
        edge=('source','target','rope')
        for t in (1,5):
            c.rope_climber=RopeClimber('target');c.rope_edge=edge
            o=observation(t);o.motion_valid=False
            self.assertFalse(c.decide(o,t).keys)
        self.assertEqual(c.failure_counts[edge[:2]],2)
        c.expire_failures(100)
        self.assertIn(edge[:2],c.failures)

    def test_rope_outside_launch_platform_is_not_a_route(self):
        ps=[Platform('source',0,184,300),Platform('target',0,280,150)]
        m=MotionProfile(calibrated=True,jump_height=90,jump_distance=100)
        self.assertNotIn(('target','rope'),graph(ps,[Rope(215,150,300)],m)['source'])


class DecisionJournalTests(unittest.TestCase):
    def test_full_history_survives_window_and_keeps_late_annotations(self):
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'frames.jsonl';m=Metrics(max_rows=2,journal=path)
            for i in range(5):
                p=SimpleNamespace(id=i,started=m.started+i,finished=m.started+i+.01)
                m.add(p,p.finished,Decision(),'test',False,0)
                m.rows[-1]['rope']={'attempts':i}
            m.close();m.close()
            rows=[json.loads(l) for l in path.read_text().splitlines()]
            self.assertEqual(len(m.rows),2)
            self.assertEqual([r['frame_id'] for r in rows],list(range(5)))
            self.assertEqual([r['rope']['attempts'] for r in rows],list(range(5)))


if __name__=='__main__':unittest.main()
