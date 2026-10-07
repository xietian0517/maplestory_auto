"""Captured live failures: static HUD camera anchors and partial identity."""
import json
from pathlib import Path
import unittest
import cv2
import numpy as np
from autofarm.realtime.model import Scene
from autofarm.realtime.perception import GroundedVision

FIXTURES=Path(__file__).parent/'fixtures/realtime'


class LiveFailureFrames(unittest.TestCase):
    def setUp(self):
        cv2.setNumThreads(2)
        d=json.loads((FIXTURES/'live_scene.json').read_text(encoding='utf-8'))
        self.seed=cv2.imread(str(FIXTURES/'live_seed.png'))
        self.image=cv2.imread(str(FIXTURES/'live_hud_occlusion.png'))
        self.vision=GroundedVision(Scene.parse(d,d['request_id'],1366,768),self.seed)
        self.vision.appearance.poses=list(np.load(FIXTURES/'live_identity_poses.npy',allow_pickle=False))

    def test_fixed_hotbar_features_cannot_override_scrolling_terrain(self):
        offset=self.vision.camera.relocalize(self.image)
        self.assertIsNotNone(offset)
        self.assertAlmostEqual(offset[0],-308,delta=2)
        self.assertAlmostEqual(offset[1],0,delta=2)
        self.assertIsNone(self.vision.camera.relocalize(np.zeros_like(self.image)))

    def test_hidden_name_and_torso_need_two_visible_head_face_observations(self):
        for confirmed in (False,True):
            actor=self.vision._player(self.image,-308,0)
            self.assertIsNotNone(actor)
            self.assertEqual(self.vision.identity_pending,not confirmed)
            self.assertAlmostEqual(actor.box.cx,1344,delta=5)
            self.assertAlmostEqual(actor.box.y2,635,delta=4)
        self.assertIsNone(self.vision._player(np.zeros_like(self.image),-308,0))

    def test_two_matching_visible_heads_are_ambiguous(self):
        image=np.zeros_like(self.seed)
        pose=self.vision.appearance.poses[0][6:38,8:44]
        image[300:332,300:336]=pose;image[301:333,601:637]=pose
        self.assertIsNone(self.vision.appearance.occluded_head(image,(0,0,1366,694)))

    def test_hotbar_occlusion_with_changed_background_needs_two_observations(self):
        image=cv2.imread(str(FIXTURES/'live_hotbar_idle.png'))
        for confirmed in (False,True):
            actor=self.vision._player(image,0,0)
            self.assertIsNotNone(actor)
            self.assertEqual(self.vision.identity_pending,not confirmed)
            self.assertAlmostEqual(actor.box.cx,1346,delta=4)
            self.assertAlmostEqual(actor.box.y2,635,delta=4)

    def test_damage_effect_over_hat_is_not_treated_as_visible_identity(self):
        image=cv2.imread(str(FIXTURES/'live_hotbar_hit.png'))
        self.assertIsNone(self.vision.appearance.occluded_head(image,(0,0,1366,694)))

    def test_other_players_remain_unmatched_without_the_reviewed_head(self):
        image=cv2.imread(str(FIXTURES/'live_hotbar_idle.png'))
        image[550:615,1300:1366]=0
        self.assertIsNone(self.vision.appearance.occluded_head(image,(0,0,1366,694)))

if __name__=='__main__':unittest.main()
