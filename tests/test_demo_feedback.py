"""Reward correctness: unknowns and replay data must not become success."""
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from autofarm.realtime.demo_feedback import VerifiedHUDReader, compare_runs, experience_delta


class FeedbackTests(unittest.TestCase):
    def test_new_session_starting_exp_does_not_change_net_reward(self):
        for start in (10134, 50000, 90000):
            with self.subTest(start=start):
                delta=experience_delta(dict(t=0,level=44,exp=start),dict(t=600,level=44,exp=start+23283))
                self.assertEqual(delta['net_exp'],23283)

    def test_exp_loss_is_preserved(self):
        result = experience_delta(dict(t=1, level=44, exp=200), dict(t=61, level=44, exp=100))
        self.assertEqual(result['net_exp'], -100)
        self.assertEqual(result['exp_per_minute'], -100)
        self.assertEqual(result['reason'], 'net_loss_review_required')

    def test_unknown_and_level_rollover_do_not_become_reward(self):
        a = dict(t=1, level=44, exp=400000)
        for b in [dict(t=2, level=45, exp=100), dict(t=2, level=None, exp=500000), dict(t=2, level=44, exp=None)]:
            with self.subTest(after=b):
                self.assertIsNone(experience_delta(a,b)['net_exp'])

    def test_time_gap_is_unknown(self):
        a=dict(t=1,level=44,exp=100);b=dict(t=3,level=44,exp=200)
        self.assertIsNone(experience_delta(a,b,max_gap=.6)['net_exp'])
        self.assertIsNone(experience_delta(a,a)['net_exp'])

    def test_human_return_cannot_be_relabelled_as_policy_return(self):
        human=dict(seconds=600,net_exp=23283)
        for mode in ['OFFLINE_REPLAY', 'OFFLINE_HUMAN_EVIDENCE', 'SHADOW_PREDICTIONS']:
            result=compare_runs(human,dict(mode=mode,seconds=600,net_exp=999999,independent_session=True,same_conditions=True,manual_interventions=0))
            self.assertFalse(result['matched_human_exp_rate'])
            self.assertIsNone(result['score'])

    def test_live_run_needs_matching_conditions_and_no_intervention(self):
        human=dict(seconds=600,net_exp=23283)
        run=dict(mode='LIVE_AUTOMATIC_OBSERVED',seconds=600,net_exp=24000,independent_session=True,same_conditions=True,manual_interventions=0)
        self.assertTrue(compare_runs(human,run)['matched_human_exp_rate'])
        for change in [dict(same_conditions=False),dict(independent_session=False),dict(manual_interventions=1),dict(net_exp=None),dict(seconds=20)]:
            self.assertFalse(compare_runs(human,{**run,**change})['matched_human_exp_rate'])

    def test_changed_hud_glyph_or_level_stays_unknown(self):
        # Tiny deterministic fixture, no font engine or production screenshots.
        im=np.zeros((20,80,3),np.uint8)
        one=np.array([[0,1],[1,1],[0,1],[0,1],[0,1]],np.uint8)
        dot=np.array([[1]],np.uint8)
        percent=np.array([[1,0,1],[0,1,0],[1,0,1]],np.uint8)
        positions=[(2,one),(15,one),(20,dot),(24,one),(30,one),(36,percent)]
        for x,glyph in positions:
            im[2:2+len(glyph),x:x+glyph.shape[1]]=glyph[:,:,None]*255
        im[1:8,10]=(0,200,50);im[1:8,42]=(0,200,50)
        im[12:18,2:8]=(10,70,220)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);cv2.imwrite(str(path/'anchor.png'),im)
            config=dict(level=44,text_roi=[0,0,50,10],level_roi=[2,12,8,18],anchors=[dict(image='anchor.png',text='11.11%')])
            reader=VerifiedHUDReader(path,config)
            self.assertEqual(reader.read(im)['exp'],1)
            changed=im.copy();changed[2,2]=255
            self.assertIsNone(reader.read(changed)['exp'])
            changed=im.copy();changed[12:18,2:8]=0
            self.assertIsNone(reader.read(changed)['level'])
            self.assertIsNone(reader.read(im[:10])['exp'])

    def test_scored_session_level_reference_replaces_the_human_level(self):
        # The glyph font is shared, but the level pixels belong to one session.
        im=np.zeros((20,80,3),np.uint8)
        one=np.array([[0,1],[1,1],[0,1],[0,1],[0,1]],np.uint8)
        dot=np.array([[1]],np.uint8)
        percent=np.array([[1,0,1],[0,1,0],[1,0,1]],np.uint8)
        for x,glyph in [(2,one),(15,one),(20,dot),(24,one),(30,one),(36,percent)]:
            im[2:2+len(glyph),x:x+glyph.shape[1]]=glyph[:,:,None]*255
        im[1:8,10]=(0,200,50);im[1:8,42]=(0,200,50)
        im[12:18,2:8]=(10,70,220)
        levelled=im.copy();levelled[12:18,2:8]=(30,220,40)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);cv2.imwrite(str(path/'anchor.png'),im);cv2.imwrite(str(path/'session.png'),levelled)
            config=dict(level=None,text_roi=[0,0,50,10],level_roi=[2,12,8,18],
                        anchors=[dict(image='anchor.png',text='11.11%')],
                        level_reference=str(path/'session.png'))
            reader=VerifiedHUDReader(path,config)
            self.assertEqual(reader.read(levelled)['level'],'session')
            self.assertEqual(reader.read(levelled)['exp'],1)
            self.assertIsNone(reader.read(im)['level'])  # Human-session level is not this run's.
            with self.assertRaises(ValueError):
                VerifiedHUDReader(path,{**config,'level_reference':'missing.png'})
            cv2.imwrite(str(path/'other.png'),im[:10])
            with self.assertRaises(ValueError):
                VerifiedHUDReader(path,{**config,'level_reference':'other.png'})


if __name__=='__main__':
    unittest.main()
