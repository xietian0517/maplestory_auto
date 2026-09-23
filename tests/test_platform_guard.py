"""Calibrated two sided detection/controller and template persistence without game input."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from unittest import TestCase, mock

import cv2
import numpy as np
from PIL import Image

from autofarm.platform_guard import GuardScene, GuardController, GuardObservation, validate_profile, profile_asset
from autofarm.vision import Hit
from guard_template_gui import save_calibration, append_monster


class GuardTests(TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        rng = np.random.default_rng(42)
        self.frame = np.full((220, 520, 3), 30, dtype=np.uint8)
        self.boxes = dict(anchor=(245, 25, 275, 45), player=(225, 90, 275, 106),
                          platform=(120, 130, 380, 150), home=(220, 115, 280, 125), observe=(20, 60, 500, 150))
        for key in ('anchor', 'player'):
            x1, y1, x2, y2 = self.boxes[key]
            self.frame[y1:y2, x1:x2] = rng.integers(0, 255, (y2-y1, x2-x1, 3), dtype=np.uint8)
        monster = rng.integers(0, 255, (30, 25, 3), dtype=np.uint8)
        self.frame[90:120, 45:70] = monster
        self.frame[90:120, 420:445] = monster
        self.path = save_calibration(Image.fromarray(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)),
            self.boxes, [(45, 90, 70, 120)], 'test', root=self.temp.name)
        self.scene = GuardScene(str(self.path))
        self.o = GuardObservation(Hit(250, 98, 1), Hit(260, 35, 1), Hit(150, 110, 1), Hit(350, 110, 1))

    def test_scene_finds_two_sides_and_camera_scroll_follows_anchor(self):
        for dx, dy in [(0, 0), (15, -10), (-15, 40)]:
            frame = cv2.warpAffine(self.frame, np.float32([[1, 0, dx], [0, 1, dy]]), (520, 220), borderValue=(30,30,30))
            o = self.scene.observe(frame)
            self.assertFalse(o.reason)
            self.assertIsNotNone(o.left)
            self.assertIsNotNone(o.right)
            self.assertAlmostEqual(o.player.x-o.anchor.x, -10)

    def test_missing_player_or_anchor_stops(self):
        for key in ('player', 'anchor'):
            frame = self.frame.copy()
            x1, y1, x2, y2 = self.boxes[key]
            frame[y1:y2, x1:x2] = 30
            self.assertTrue(self.scene.observe(frame).reason)

    def test_monsters_outside_region_or_range_do_not_trigger(self):
        self.scene.attack_range = 80
        o = self.scene.observe(self.frame)
        self.assertIsNone(o.left)
        self.assertIsNone(o.right)

    def test_changed_player_height_is_accepted_but_edge_is_not(self):
        crop = self.frame[90:106, 225:275].copy()
        for x, y, stopped in [(225, 175, False), (375, 90, True)]:
            frame = self.frame.copy()
            frame[90:106, 225:275] = 30
            frame[y:y+16, x:x+50] = crop
            self.assertEqual(bool(self.scene.observe(frame).reason), stopped)

    def test_right_and_left_use_corresponding_motion_and_attack_key(self):
        for side, dx in [('right', 6), ('left', -6)]:
            c = GuardController(self.scene.data, side, 'a')
            self.assertIsNone(c.decide(self.o, 0))
            action = c.decide(self.o, .01)
            self.assertEqual(action[0], side)
            c.applied(action[0], self.o, .01)
            moved = replace(self.o, player=Hit(250+dx, 190, 1))
            self.assertIsNone(c.decide(moved, .06))
            self.assertEqual(c.decide(moved, .16)[0], 'a')
            no_monster = replace(moved, **{side: None})
            self.assertIsNone(c.decide(no_monster, .3))

    def test_both_sides_stick_until_current_target_is_gone(self):
        c = GuardController(self.scene.data, 'both')
        for now in (0, .1, 1):
            c.decide(self.o, now)
            self.assertEqual(c.side, 'right')
        left_only = replace(self.o, right=None)
        c.decide(left_only, 2)
        self.assertEqual(c.side, 'right')
        c.decide(left_only, 2.4)
        self.assertEqual(c.side, 'left')
        for now in (3, 4, 5):
            c.decide(self.o, now)
            self.assertEqual(c.side, 'left')

    def test_ignored_turn_or_missing_recognition_never_attacks(self):
        c = GuardController(self.scene.data, 'left', 'a')
        for i in range(20):
            action = c.decide(self.o, i*.04)
            if action:
                c.applied(action[0], self.o, i*.04)
                self.assertNotEqual(action[0], 'a')
        self.assertIsNone(c.decide(replace(self.o, reason='missing'), 1))

    def test_append_monster_and_profile_survive_reload(self):
        image = Image.fromarray(self.frame[90:120, 45:70])
        append_monster(str(self.path), image)
        scene = GuardScene(str(self.path))
        self.assertEqual(len(scene.monsters), 4)
        self.assertEqual(scene.data['name'], 'test')

    def test_invalid_geometry_and_external_asset_are_rejected(self):
        for change in [dict(target_tolerance=10), dict(safe_left=240), dict(max_speed=float('nan'))]:
            with self.assertRaises(ValueError):
                validate_profile(self.scene.data | change)
        with self.assertRaises(ValueError):
            profile_asset(self.path, '../../outside.png')
        with self.assertRaises(ValueError):
            save_calibration(Image.fromarray(self.frame), {}, [], 'bad', root=self.temp.name)

    def test_calibration_and_monster_dialog_save_real_assets(self):
        import tkinter as tk
        from guard_template_gui import GuardCalibrationDialog, MonsterCropDialog
        app = tk.Tk()
        app.withdraw()
        self.addCleanup(app.destroy)
        saved = []
        image = Image.fromarray(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB))
        append_monster(str(self.path), image.crop((420, 90, 445, 120)))
        data = json.loads(self.path.read_text(encoding='utf-8')) | {'_profile_path': str(self.path)}
        dialog = GuardCalibrationDialog(app, image, saved.append, existing=data)
        dialog.withdraw()
        with mock.patch('guard_template_gui.V.program_dir', return_value=Path(self.temp.name)):
            dialog.save()
        self.assertEqual(len(saved), 1)
        self.assertTrue(GuardScene(str(saved[0])).observe(self.frame).left)
        self.assertEqual(len(GuardScene(str(saved[0])).data['monster_templates']), 2)
        crops = []
        crop = MonsterCropDialog(app, image.crop((45, 90, 70, 120)), crops.append)
        crop.withdraw()
        crop.save()
        self.assertEqual(crops[0].size, (25, 30))

    def test_guard_runtime_releases_before_buff_and_preserves_state(self):
        from autofarm.platform_guard import run
        from autofarm.bot import BotStopped
        from farm import Config
        bot, scene, controller, inputs, buffs = (mock.MagicMock() for _ in range(5))
        scene.data = self.scene.data
        scene.observe.return_value = self.o
        controller.decide.return_value = ('shift', .4, 'attack')
        controller.can_buff.return_value = True
        inputs.__enter__.return_value = inputs
        inputs.epoch, inputs.key = 0, 'shift'
        bot.gate.side_effect = [False, BotStopped()]
        buffs.cooling.return_value = False
        buffs.due.return_value = buffs.prepare.return_value = True
        order = []
        inputs.clear.side_effect = lambda: order.append('release')
        buffs.cast_one.side_effect = lambda *a, **kw: order.append('buff') or True
        with mock.patch('autofarm.platform_guard.GuardScene', return_value=scene), \
             mock.patch('autofarm.platform_guard.GuardController', return_value=controller), \
             mock.patch('autofarm.held_input.HeldInput', return_value=inputs), \
             mock.patch('autofarm.buffs.BuffScheduler', return_value=buffs), \
             mock.patch('autofarm.platform_guard.V.capture'), \
             mock.patch('autofarm.platform_guard.time.monotonic', return_value=10):
            with self.assertRaises(BotStopped):
                run(bot, Config())
        self.assertEqual(order, ['release', 'buff'])
        inputs.apply.assert_not_called()
        controller.motion_stopped.assert_called_once()
