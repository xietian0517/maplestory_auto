import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts.analyze_live_evaluation import analyze


class BenchmarkBoundaryTests(unittest.TestCase):
    def test_parking_experience_cannot_inflate_farming_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);demo=root/'demo';analysis=demo/'analysis';analysis.mkdir(parents=True)
            (analysis/'hud_calibration.json').write_text('{}')
            (analysis/'human_baseline.json').write_text(json.dumps(dict(hud_seconds=600,exp_per_minute_hud=1000,net_exp=10000)))
            run=root/'run';run.mkdir()
            rows=[dict(t=t,image=str(exp),exp=exp,level=44) for t,exp in
                  [(0.1,100),(0.3,100),(0.8,120),(0.9,120),(1.1,1000),(1.3,1000)]]
            (run/'hud.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            (run/'intervals.json').write_text(json.dumps(dict(farming_ended_at=.9)))
            with patch('scripts.analyze_live_evaluation.VerifiedHUDReader') as reader, patch('scripts.analyze_live_evaluation.cv2.imread'):
                reader.return_value.read.return_value={}
                result=analyze(run,demo)
            self.assertEqual(result['net_exp'],20)
            self.assertEqual(result['hud_samples'],4)
            self.assertEqual(result['last']['t'],.9)
            self.assertTrue(result['parking_excluded'])
            self.assertFalse(result['human_comparison']['matched_human_exp_rate'])
            self.assertIsNone(result['effective_damage'])
            self.assertEqual(result['mode'],'UNVERIFIED_RECORDING')
            self.assertFalse(result['independent_session'])

    def test_human_recording_cannot_be_relabelled_live_by_cli_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);demo=root/'demo';analysis=demo/'analysis';analysis.mkdir(parents=True)
            (analysis/'hud_calibration.json').write_text('{}')
            (analysis/'human_baseline.json').write_text(json.dumps(dict(hud_seconds=600,exp_per_minute_hud=1000,net_exp=10000)))
            (demo/'session.json').write_text(json.dumps(dict(mode='HUMAN_DEMONSTRATION_READ_ONLY',monotonic_origin=123)))
            rows=[dict(t=t,image=str(exp),exp=exp,level=44) for t,exp in [(0,100),(0.2,100),(600,20100),(600.2,20100)]]
            (demo/'hud.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            with patch('scripts.analyze_live_evaluation.VerifiedHUDReader') as reader, patch('scripts.analyze_live_evaluation.cv2.imread'):
                reader.return_value.read.return_value={}
                result=analyze(demo,demo,matched_conditions=True,no_manual=True)
            self.assertEqual(result['net_exp'],20000)
            self.assertFalse(result['independent_session'])
            self.assertFalse(result['human_comparison']['matched_human_exp_rate'])

    def test_live_flags_cannot_replace_missing_input_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);demo=root/'demo';analysis=demo/'analysis';analysis.mkdir(parents=True)
            (analysis/'hud_calibration.json').write_text('{}')
            (analysis/'human_baseline.json').write_text(json.dumps(dict(hud_seconds=600,exp_per_minute_hud=1000,net_exp=10000)))
            run=root/'run'/'evaluation';config=run/'configuration_snapshot';config.mkdir(parents=True)
            (run/'session.json').write_text(json.dumps(dict(mode='LIVE_AUTOMATIC_OBSERVED',monotonic_origin=123)))
            (config/'run_configuration.json').write_text(json.dumps(dict(live=True)))
            (run.parent/'report.json').write_text(json.dumps(dict(mode='LIVE',keys_released=True)))
            rows=[dict(t=t,image=str(exp),exp=exp,level=44) for t,exp in [(0,100),(.2,100),(600,20100),(600.2,20100)]]
            (run/'hud.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            with patch('scripts.analyze_live_evaluation.VerifiedHUDReader') as reader, patch('scripts.analyze_live_evaluation.cv2.imread'):
                reader.return_value.read.return_value={}
                result=analyze(run,demo,matched_conditions=True,no_manual=True)
            self.assertTrue(result['live_provenance_confirmed'])
            self.assertIsNone(result['manual_interventions'])
            self.assertEqual(result['input_audit']['reason'],'audit_missing')
            self.assertFalse(result['human_comparison']['matched_human_exp_rate'])


if __name__=='__main__':unittest.main()
