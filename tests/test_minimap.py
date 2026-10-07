"""Minimap scrolling, identity ambiguity, alignment and navigation regressions."""
from pathlib import Path
import unittest

import cv2
import numpy as np

from autofarm.realtime.minimap import (MinimapNavigator, locate_minimap,
                                      terrain_translation, yellow_points)
from autofarm.realtime.model import Actor, Box, Observation, Platform
from autofarm.realtime.control import Controller


FIXTURES = Path(__file__).parent/'fixtures'/'realtime'


def actor(x, y):
    return Actor(Box(x-15, y-48, x+15, y), .99)


class MinimapTests(unittest.TestCase):
    def setUp(self):
        cv2.setNumThreads(2)
        self.seed = cv2.imread(str(FIXTURES/'seed.png'))

    def test_actual_minimap_and_yellow_player(self):
        roi = locate_minimap(self.seed)
        self.assertIsNotNone(roi)
        x, y, w, h = roi
        self.assertLess(abs(x-6)+abs(y-72), 4)
        points = yellow_points(self.seed[y:y+h, x:x+w])
        self.assertEqual(len(points), 1)
        self.assertAlmostEqual(points[0][0]+x, 59.5, delta=1)

    def test_user_screenshot_annotation_does_not_break_detection(self):
        image = cv2.imread(str(FIXTURES/'minimap_reference.png'))
        roi = locate_minimap(image)
        self.assertIsNotNone(roi)
        self.assertLess(abs(roi[0]-10)+abs(roi[1]-111), 4)

    def test_real_camera_scroll_is_not_minimap_terrain_scroll(self):
        x, y, w, h = locate_minimap(self.seed)
        moved = cv2.imread(str(FIXTURES/'camera_scroll.png'))
        shift = terrain_translation(self.seed[y:y+h, x:x+w], moved[y:y+h, x:x+w])
        self.assertIsNotNone(shift)
        np.testing.assert_allclose(shift, (0, 0), atol=1)

    def test_static_patches_track_horizontal_vertical_and_diagonal_scroll(self):
        x, y, w, h = locate_minimap(self.seed)
        pane = self.seed[y:y+h, x:x+w]
        for dx, dy in ((-7, 0), (0, -8), (6, -5)):
            with self.subTest(shift=(dx, dy)):
                moved = cv2.warpAffine(pane, np.float32([[1, 0, dx], [0, 1, dy]]), (w, h))
                shift = terrain_translation(pane, moved)
                self.assertIsNotNone(shift)
                np.testing.assert_allclose(shift, (dx, dy), atol=.5)

    def test_unrelated_or_zoomed_terrain_cannot_be_stitched(self):
        x, y, w, h = locate_minimap(self.seed)
        pane = self.seed[y:y+h, x:x+w]
        self.assertIsNone(terrain_translation(pane, np.zeros_like(pane)))
        zoomed = cv2.resize(pane, None, fx=1.4, fy=1.4)[:h, :w]
        self.assertIsNone(terrain_translation(pane, zoomed))

    def test_yellow_dot_is_not_used_as_scroll_anchor(self):
        x, y, w, h = locate_minimap(self.seed)
        pane = self.seed[y:y+h, x:x+w].copy()
        moved = pane.copy()
        moved[58:68, 49:59] = (40, 45, 40)
        cv2.circle(moved, (43, 63), 2, (0, 255, 255), -1)
        np.testing.assert_allclose(terrain_translation(pane, moved), (0, 0), atol=.5)

    def test_ambiguous_marker_disables_guidance(self):
        m = MinimapNavigator()
        for i in range(3): m.observe(self.seed, 1+i*.04)
        x, y, _, _ = m.roi
        changed = self.seed.copy()
        cv2.circle(changed, (x+15, y+65), 2, (0, 255, 255), -1)
        m.observe(changed, 1.12)
        self.assertEqual(m.reason, 'marker_ambiguous')
        self.assertIsNone(m.position)
        self.assertIsNone(m.player_hint((0, 0), 1.12))

    def test_hidden_map_discards_old_atlas_and_reacquires(self):
        m = MinimapNavigator()
        for i in range(4): m.observe(self.seed, 1+i*.04)
        generation = m.generation
        self.assertTrue(m.platforms)
        x, y, w, h = m.roi
        hidden = self.seed.copy(); hidden[y:y+h, x:x+w] = 0
        m.observe(hidden, 1.16)
        self.assertIsNone(m.position)
        self.assertFalse(m.platforms)
        self.assertGreater(m.generation, generation)
        for i in range(3): m.observe(self.seed, 2.5+i*.04)
        self.assertIsNotNone(m.position)

    def test_paused_capture_does_not_reuse_alignment(self):
        m = MinimapNavigator()
        for i in range(3): m.observe(self.seed, 1+i*.04)
        m.scale = .06; m.intercept = np.zeros(2); m.fit_at = 1.08
        m.observe(self.seed, 3)
        self.assertIsNone(m.scale)
        self.assertIsNone(m.position)

    def test_scroll_is_removed_from_player_world_position(self):
        x, y, w, h = locate_minimap(self.seed)
        pane = self.seed[y:y+h, x:x+w]
        m = MinimapNavigator()
        for i in range(3): m.observe(self.seed, 1+i*.04)
        original = m.position.copy()
        for i, (dx, dy) in enumerate(((-3, -3), (-6, -6))):
            im = self.seed.copy()
            im[y:y+h, x:x+w] = cv2.warpAffine(pane, np.float32([[1, 0, dx], [0, 1, dy]]), (w, h))
            m.observe(im, 1.12+i*.04)
            np.testing.assert_allclose(m.position, original, atol=.5)
        np.testing.assert_allclose(m.scroll, (-6, -6), atol=.5)

    def test_scale_uses_main_world_motion_not_screen_motion(self):
        m = MinimapNavigator()
        for i in range(12):
            # Player stays in the screen center while both cameras move.
            world = np.array([300+i*20, 400.])
            m.position = world*.06 + np.array([10, 20])
            m.align(actor(300, 400), (-i*20, 0), 1+i*.04)
        self.assertAlmostEqual(m.scale, .06, places=5)
        self.assertTrue(m.ready(1.44))
        np.testing.assert_allclose(m.player_hint((-220, 0), 1.44), (300, 400))
        m.position += (10, 0)
        m.align(actor(300, 400), (-220, 0), 1.48)
        self.assertIsNone(m.scale)

    def test_scrolling_video_with_centered_marker_learns_world_scale(self):
        x, y, w, h = locate_minimap(self.seed)
        pane = self.seed[y:y+h, x:x+w].copy()
        mask = np.zeros((h, w), np.uint8); mask[58:68, 49:59] = 255
        texture = cv2.inpaint(pane, mask, 3, cv2.INPAINT_TELEA)
        m = MinimapNavigator(); m.roi = (x, y, w, h)
        for i in range(16):
            im = self.seed.copy()
            shifted = cv2.warpAffine(texture, np.float32([[1, 0, -i], [0, 1, 0]]), (w, h))
            cv2.circle(shifted, (54, 63), 2, (0, 255, 255), -1)
            im[y:y+h, x:x+w] = shifted
            t = 1+i*.08
            m.observe(im, t)
            m.align(actor(300, 400), (-i*20, 0), t)
        self.assertTrue(m.ready(t), m.status())
        self.assertAlmostEqual(m.scale, .05, delta=.002)
        np.testing.assert_allclose(m.scroll, (-15, 0), atol=.5)
        np.testing.assert_allclose(m.player_hint((-300, 0), t), (300, 400), atol=2)

    def test_stationary_player_cannot_calibrate_scale(self):
        m = MinimapNavigator()
        for i in range(20):
            m.position = np.array([30., 40.])
            m.align(actor(300, 400), (0, 0), 1+i*.04)
        self.assertIsNone(m.scale)

    def test_guidance_returns_only_observed_progress_toward_frontier(self):
        m = MinimapNavigator()
        m.scale = .06; m.intercept = np.zeros(2); m.fit_at = 1
        m.position = np.array([12., 18.]); m.reason = 'tracking'
        m.platforms = [(20, 40, 30)]
        platforms = [Platform('here', 0, 400, 300), Platform('next', 250, 450, 400),
                     Platform('away', 0, 100, 200), Platform('offscreen', 1500, 1800, 400)]
        before = list(platforms)
        goals = m.goals(platforms, actor(200, 300), (0, 0), 1, Box(0, 0, 1366, 693))
        self.assertEqual(goals, ['next'])
        self.assertEqual(platforms, before)

    def test_minimap_goal_still_requires_a_main_screen_route(self):
        o = Observation(1, 1, actor(150, 300), platforms=[Platform('a', 0, 300, 300),
                        Platform('b', 0, 300, 400)], minimap_goals=['invented', 'b'])
        c = Controller(navigate=True); c.decide(o, 1)
        o.captured_at = 2.6
        d = c.decide(o, 2.6)
        self.assertEqual(d.target, 'b')
        self.assertEqual(d.reason, 'drop')

    def test_disable_has_no_observations_or_goals(self):
        m = MinimapNavigator(enabled=False); m.observe(self.seed, 1)
        self.assertEqual(m.reason, 'disabled')
        self.assertIsNone(m.roi)
        self.assertFalse(m.ready(1))


if __name__ == '__main__':
    unittest.main()
