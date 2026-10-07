import json
from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from autofarm.realtime.health import HealthGuard


class HealthTests(unittest.TestCase):
    def setUp(self):
        root=Path(__file__).parent/'fixtures/realtime'
        self.config=json.loads((root/'health_calibration.json').read_text())
        self.guard=HealthGuard(self.config,self.config['request_id'],1366,768)
        self.image=np.zeros((768,1366,3),np.uint8)
        self.image[732:744,485:627]=cv2.imread(str(root/'health_1165.png'))

    def test_reads_reviewed_real_hp_pixels(self):
        self.assertEqual(self.guard.read(self.image),(1165,'exact_calibrated_glyphs'))
        self.assertIsNone(self.guard.observe(self.image,1))
        changed=self.image.copy();changed[734:741,491:493]=0
        self.assertIsNone(self.guard.read(changed)[0])

    def test_binding_and_maximum_mismatch_remain_unknown(self):
        with self.assertRaises(ValueError):HealthGuard(self.config,'other',1366,768)
        with self.assertRaises(ValueError):HealthGuard(self.config,self.config['request_id'],800,600)
        self.guard.maximum=1586
        self.assertEqual(self.guard.read(self.image)[1],'health_range_unknown')
        self.assertIsNone(self.guard.read(self.image[:700])[0])

    def changed_maximum_image(self):
        image=self.image.copy()
        image[732:744,485:627]=cv2.imread(str(Path(__file__).parent/'fixtures/realtime/health_1503_1682.png'))
        return image

    def test_current_hud_confirms_new_maximum_without_automatic_return(self):
        image=self.changed_maximum_image();self.guard.maximum=1634
        self.assertEqual(self.guard.read(image),(None,'health_range_unknown'))
        for t in (1,1.04):
            self.assertIsNone(self.guard.observe(image,t));self.assertIsNone(self.guard.hp)
            self.assertEqual(self.guard.maximum,1634)
        self.assertIsNone(self.guard.observe(image,1.11))
        self.assertEqual((self.guard.hp,self.guard.maximum),(1503,1682))
        for i in range(30):self.assertIsNone(self.guard.observe(image,1.2+i*.1))
        self.assertIsNone(self.guard.reason)

    def test_maximum_confirmation_resets_on_gap_or_unknown_glyphs(self):
        image=self.changed_maximum_image()
        self.guard.observe(image,1);self.guard.observe(image,1.04)
        self.guard.observe(image,2)
        self.assertEqual(self.guard.maximum_samples,1)
        self.guard.observe(self.image*0,2.04)
        self.assertIsNone(self.guard.maximum_candidate)
        self.guard.observe(image,2.08);self.guard.observe(image,2.12)
        self.assertEqual(self.guard.maximum,1610)
        self.guard.observe(image,2.2);self.assertEqual(self.guard.maximum,1682)

    def test_updated_maximum_still_triggers_confirmed_low_or_zero_hp(self):
        for current,reason in ((500,'low_health_return'),(0,'health_depleted_stop')):
            with self.subTest(current=current):
                guard=HealthGuard(self.config,self.config['request_id'],1366,768)
                with patch.object(guard,'read_numbers',return_value=(current,1682,'exact_calibrated_glyphs')):
                    for t in (1,1.04,1.11,1.16,1.23):guard.observe(self.image,t)
                self.assertEqual(guard.maximum,1682);self.assertEqual(guard.reason,reason)

    def test_alternating_maxima_never_replace_calibration(self):
        with patch.object(self.guard,'read_numbers') as read:
            for i in range(30):
                read.return_value=(1500,1682 if i%2 else 1706,'exact_calibrated_glyphs')
                self.guard.observe(self.image,1+i*.1)
        self.assertEqual(self.guard.maximum,1610)
        self.assertEqual(self.guard.reason,'health_unreadable_return')

    def test_low_requires_neighbouring_samples_and_stays_latched(self):
        with patch.object(self.guard,'read',return_value=(600,'exact_calibrated_glyphs')):
            self.assertIsNone(self.guard.observe(self.image,1))
            self.assertIsNone(self.guard.observe(self.image,1.04))
            self.assertEqual(self.guard.observe(self.image,1.11),'low_health_return')
        with patch.object(self.guard,'read',return_value=(1610,'exact_calibrated_glyphs')):
            self.assertEqual(self.guard.observe(self.image,1.2),'low_health_return')

    def test_unknown_does_not_become_zero_or_count_as_low_sample(self):
        with patch.object(self.guard,'read',return_value=(600,'exact_calibrated_glyphs')):
            self.guard.observe(self.image,1);self.guard.observe(self.image,1.04)
        with patch.object(self.guard,'read',return_value=(None,'glyph_unknown')):
            self.assertIsNone(self.guard.observe(self.image,1.1));self.assertEqual(self.guard.low_samples,0)
            for i in range(1,21):self.guard.observe(self.image,1.1+i*.1)
            self.assertEqual(self.guard.reason,'health_unreadable_return');self.assertIsNone(self.guard.hp)

    def test_gap_or_healthy_sample_resets_low_sequence(self):
        with patch.object(self.guard,'read',return_value=(600,'exact_calibrated_glyphs')):
            self.guard.observe(self.image,1);self.guard.observe(self.image,1.04)
            self.assertIsNone(self.guard.observe(self.image,2));self.assertEqual(self.guard.low_samples,1)
        self.guard.observe(self.image,2.1)
        self.assertEqual(self.guard.low_samples,0)

    def test_confirmed_zero_requests_release_without_respawn_or_navigation(self):
        with patch.object(self.guard,'read',return_value=(0,'exact_calibrated_glyphs')):
            for t in (1,1.05,1.11):self.guard.observe(self.image,t)
        self.assertEqual(self.guard.reason,'health_depleted_stop')

    def test_pet_recovery_mode_records_low_and_unknown_without_return(self):
        self.config['return_enabled']=False
        g=HealthGuard(self.config,self.config['request_id'],1366,768)
        for value in (600,None,150):
            with patch.object(g,'read',return_value=(value,'test')):
                for i in range(30):self.assertIsNone(g.observe(self.image,1+i*.1))
                self.assertEqual(g.low_samples,0);self.assertEqual(g.hp,value)
        self.assertFalse(g.status()['return_enabled']);self.assertIsNone(g.status()['minimum_fraction'])


if __name__=='__main__':unittest.main()
