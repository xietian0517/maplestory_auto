import json
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from autofarm.realtime.live_evaluation import LiveEvaluation
from autofarm.realtime.control import standing_platform
from autofarm.realtime.model import Actor, Box, Observation, Platform


class LiveEvidenceTests(unittest.TestCase):
    def test_packaged_recorder_snapshots_executable_instead_of_missing_python_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            exe=Path(tmp)/'app.exe';exe.write_bytes(b'test-build')
            folder=Path(tmp)/'run';folder.mkdir()
            with patch('autofarm.realtime.live_evaluation.sys.frozen',True,create=True), \
                 patch('autofarm.realtime.live_evaluation.sys.executable',str(exe)), \
                 patch('autofarm.realtime.live_evaluation.VideoSegments') as video, \
                 patch('autofarm.realtime.live_evaluation.write_review',lambda f:(f/'review.html').write_text('test')):
                recorder=video.return_value.__enter__.return_value
                recorder.written=0;recorder.dropped=0;recorder.error=None
                with LiveEvaluation(folder,object(),30):pass
            self.assertEqual((folder/'evaluation/source_snapshot/app.exe').read_bytes(),b'test-build')

    def test_input_snapshot_precedes_capture_and_includes_final_release(self):
        now=[10.]; calls=[]
        adapter=SimpleNamespace(send_key=lambda k,u:calls.append((k,u)))
        class Video:
            error=None;written=0;dropped=0
            def __init__(self,*args):self.rows=[]
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def offer(self,image,row):self.rows.append(row);self.written+=1
        with tempfile.TemporaryDirectory() as tmp, patch('autofarm.realtime.live_evaluation.VideoSegments',Video), patch('autofarm.realtime.live_evaluation.write_review',lambda f:(f/'review.html').write_text('人工示范回看',encoding='utf-8')):
            config=Path(tmp)/'run_configuration.json';config.write_text('{"seconds":600}')
            with LiveEvaluation(tmp,adapter,30,clock=lambda:now[0]) as evidence:
                config.write_text('{"seconds":1}')
                now[0]=10.2;evidence.send_key('shift',False)
                now[0]=10.4;evidence.send_key('shift',True)
                evidence.offer(SimpleNamespace(id=1,started=10.3,finished=10.31,foreground=True,image=np.zeros((2,2,3),np.uint8)))
                self.assertEqual(evidence.video.rows[0]['keys'],['shift'])
                evidence.offer(SimpleNamespace(id=2,started=10.5,finished=10.51,foreground=True,image=np.zeros((2,2,3),np.uint8)))
                self.assertEqual(evidence.video.rows[1]['keys'],[])
            events=[json.loads(x) for x in (Path(tmp)/'evaluation/inputs.jsonl').read_text().splitlines()]
            self.assertEqual([e['event'] for e in events],['key_down','key_up','recording_end'])
            self.assertTrue(events[-1]['release_observed'])
            self.assertEqual(calls,[('shift',False),('shift',True)])
            snapshot=Path(tmp)/'evaluation/configuration_snapshot/run_configuration.json'
            self.assertEqual(json.loads(snapshot.read_text())['seconds'],600)
            session=json.loads((Path(tmp)/'evaluation/session.json').read_text())
            self.assertEqual(session['configuration_sha256']['run_configuration.json'],hashlib.sha256(snapshot.read_bytes()).hexdigest())
            source=Path(tmp)/'evaluation/source_snapshot/autofarm/realtime/control.py'
            self.assertTrue(source.exists())
            self.assertEqual(session['source_sha256'][str(Path('autofarm/realtime/control.py'))],hashlib.sha256(source.read_bytes()).hexdigest())

    def test_small_edge_uncertainty_does_not_bridge_gap_or_catch_falling_player(self):
        floor=Platform('edge',657,1045,459)
        def locate(x,vy=0):
            return standing_platform(Observation(1,1,Actor(Box(x-15,410,x+15,458),1,0,vy),platforms=[floor]))
        self.assertEqual(locate(652),floor)
        self.assertEqual(locate(648),floor)  # Observed idle sprite 9px past annotated edge.
        self.assertIsNone(locate(640))
        self.assertIsNone(locate(652,120))

if __name__=='__main__':unittest.main()
