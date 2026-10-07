import json
from pathlib import Path
import tempfile
import unittest
from autofarm.realtime.input_audit import keyboard_event,assess
from autofarm.winapi import INPUT_SOURCE_TAG


class InputAuditTests(unittest.TestCase):
    def test_external_keys_are_scoped_and_redacted(self):
        self.assertIsNone(keyboard_event(65,0,0,False,1))
        self.assertEqual(keyboard_event(65,0,0,True,1)['key'],'other')
        self.assertEqual(keyboard_event(0x25,0,0,True,1)['source'],'physical')
        self.assertEqual(keyboard_event(0x25,0x10,0,True,1)['source'],'external_injected')
        self.assertEqual(keyboard_event(0x25,0x90,INPUT_SOURCE_TAG,False,1)['event'],'key_up')
        self.assertEqual(keyboard_event(0x25,0x90,INPUT_SOURCE_TAG,False,1)['source'],'controller')
        self.assertNotEqual(keyboard_event(0x25,0,INPUT_SOURCE_TAG,True,1)['source'],'controller')

    def fixture(self,p,rows,errors=None):
        info=dict(mode='WINDOWS_GAME_INPUT_AUDIT',installed=True,start=0,end=3,errors=errors or [])
        (p/'input_audit_report.json').write_text(json.dumps(info))
        (p/'input_audit.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
        s=[dict(t=1,key='left',event='key_down'),dict(t=2,key='left',event='key_up')]
        (p/'inputs.jsonl').write_text('\n'.join(json.dumps(r) for r in s))

    def own(self):
        return [keyboard_event(0x25,0x10,INPUT_SOURCE_TAG,True,.999),
                keyboard_event(0x25,0x90,INPUT_SOURCE_TAG,True,1.999)]

    def test_missing_audit_is_unknown_not_zero(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(assess(d,.5,2.5)['manual_interventions'])

    def test_all_edges_match_and_external_intervention_is_counted(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self.fixture(p,self.own())
            self.assertEqual(assess(p,.5,2.5)['manual_interventions'],0)
            self.fixture(p,self.own()+[keyboard_event(0x25,0,0,True,1.5)])
            self.assertEqual(assess(p,.5,2.5)['manual_interventions'],1)

    def test_missing_duplicate_or_delayed_edges_invalidate_coverage(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            for rows in (self.own()[:1],self.own()+[self.own()[0]],
                         [self.own()[0],dict(self.own()[1],t=2.3)]):
                self.fixture(p,rows)
                self.assertFalse(assess(p,.5,2.5)['confirmed'])
                self.assertIsNone(assess(p,.5,2.5)['manual_interventions'])

    def test_pump_failure_and_interval_outside_coverage_stay_unknown(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self.fixture(p,self.own(),['message_pump_gap'])
            self.assertFalse(assess(p,.5,2.5)['confirmed'])
            self.fixture(p,self.own());self.assertFalse(assess(p,.5,3.1)['confirmed'])

    def test_parking_inputs_and_mouse_motion_are_not_farming_interventions(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self.fixture(p,self.own()+[keyboard_event(0x25,0,0,True,2.7),
                dict(t=1.5,event='move',source='physical',foreground=True,device='mouse')])
            r=assess(p,.5,2.5)
            self.assertEqual(r['manual_interventions'],0);self.assertEqual(r['mouse_motion_events'],1)

    def test_hook_before_interval_end_and_submission_after_are_one_edge(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self.fixture(p,self.own())
            # The hook fires just before SendInput returns. A HUD endpoint can
            # lie between those two timestamps without creating an extra key.
            result=assess(p,.5,1.9995)
            self.assertTrue(result['confirmed'])
            self.assertEqual(result['submitted_edges'],1)
            self.assertEqual(result['extra_tagged_edges'],0)


if __name__=='__main__':unittest.main()
