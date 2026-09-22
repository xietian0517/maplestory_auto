"""离线检查：不连接游戏，不发送键盘输入。"""
from dataclasses import replace
from pathlib import Path
import tempfile
import json
from unittest import TestCase, mock

import cv2
import numpy as np

from autofarm.rope_archer import ArcherController, Observation, RopeScene, FailureSnapshots, run
from autofarm.version import ARCHER_VERSION
from autofarm.vision import Hit
from autofarm.bot import Bot, BotStopped
from farm import Config

PROFILE = Path(__file__).resolve().parents[1] / 'templates/rope_archer/profile.json'


class ControllerTests(TestCase):
    def setUp(self):
        self.scene = RopeScene(PROFILE)
        self.c = ArcherController(self.scene.data)
        self.o = Observation(Hit(463, 522.5, 1), Hit(472, 202, 1), Hit(900, 516, 1))

    def stable(self, o, now=0):
        self.c.decide(o, now)
        return self.c.decide(o, now)

    def test_attack_stays_held_until_monkey_disappears(self):
        self.assertEqual(self.stable(self.o)[0], 'right')
        for now in (.1, .3, .7, 1.0, 2.0):
            self.assertEqual(self.c.decide(self.o, now)[0], 'shift')
        self.assertIsNone(self.c.decide(replace(self.o, monkey=None), 2.1))

    def test_no_monkey_means_no_attack_or_turn(self):
        o = replace(self.o, monkey=None)
        self.assertIsNone(self.stable(o))
        self.assertIsNone(self.c.decide(o, 10))

    def test_knockback_continues_moving_without_stability_pauses(self):
        o = replace(self.o, player=Hit(400, 522.5, 1), monkey=None)
        key, lease, _ = self.stable(o)
        self.assertEqual(key, 'right')
        self.assertGreater(lease, .2)
        for i, x in enumerate((420, 430, 440, 450)):
            action = self.c.decide(replace(o, player=Hit(x, 522.5, 1)), .1 + i * .1)
            self.assertEqual(action[0], 'right')
        self.assertIsNone(self.c.decide(replace(o, player=Hit(458, 522.5, 1)), .6))

    def test_right_edge_returns_left_and_leaves_turning_space(self):
        o = replace(self.o, player=Hit(510, 522.5, 1))
        self.assertEqual(self.stable(o)[0], 'left')
        # 进入原来的站位范围也继续这一段回位，避免左一下右一下。
        self.assertEqual(self.c.decide(replace(o, player=Hit(495, 522.5, 1)), .1)[0], 'left')
        self.assertIsNone(self.c.decide(replace(o, player=Hit(456, 522.5, 1)), .2))
        self.assertEqual(self.c.decide(replace(o, player=Hit(456, 522.5, 1)), .4)[0], 'right')
        self.assertEqual(self.c.decide(replace(o, player=Hit(460, 522.5, 1)), .5)[0], 'shift')

    def test_arrival_noise_does_not_restart_movement(self):
        o = replace(self.o, player=Hit(420, 522.5, 1), monkey=None)
        self.stable(o)
        self.assertIsNone(self.c.decide(replace(o, player=Hit(458, 522.5, 1)), .1))
        for i, x in enumerate((455, 460, 470, 453, 457)):
            self.assertIsNone(self.c.decide(replace(o, player=Hit(x, 522.5, 1)), .4 + i * .1))

    def test_missing_player_anchor_or_wrong_floor_stops(self):
        for o in (Observation(reason='missing'), replace(self.o, player=None),
                  replace(self.o, anchor=None), replace(self.o, reason='wrong floor')):
            self.assertIsNone(self.stable(o))

    def test_attack_releases_for_knockback_recovery(self):
        self.stable(self.o)
        self.assertEqual(self.c.decide(self.o, .1)[0], 'shift')
        self.assertEqual(self.c.decide(replace(self.o, player=Hit(430, 522.5, 1)), .2)[0], 'right')

    def test_move_deadline_respects_edge_even_when_far_from_target(self):
        o = replace(self.o, player=Hit(430, 522.5, 1))
        _, lease, _ = self.stable(o)
        self.assertLessEqual(430 + lease * self.c.d['max_speed'] + self.c.reserve, 483)

    def test_left_recovery_deadline_cannot_cross_home_band(self):
        o = replace(self.o, player=Hit(510, 522.5, 1))
        key, lease, _ = self.stable(o)
        self.assertEqual(key, 'left')
        self.assertGreaterEqual(510 - lease * self.c.d['max_speed'] - self.c.reserve, 443)

    def test_guard_band_takes_priority_over_settling_and_attack(self):
        self.stable(self.o)
        self.c.settle_until = 100
        self.c.facing_right = True
        self.assertEqual(self.c.decide(replace(self.o, player=Hit(500, 522.5, 1)), .2)[0], 'left')

    def test_turn_near_guard_band_first_retreats(self):
        self.assertEqual(self.stable(replace(self.o, player=Hit(480, 522.5, 1)))[0], 'left')

    def test_airborne_small_height_change_still_blocks_motion_and_attack(self):
        o = replace(self.o, player=Hit(400, 528, 1))
        self.assertIsNone(self.stable(o))
        self.assertIsNone(self.c.decide(o, .2))

    def test_blind_hold_with_release_delay_and_position_error_stays_in_home_band(self):
        d = self.c.d
        for x in range(285, 512):
            c = ArcherController(d)
            o = replace(self.o, player=Hit(x, 522.5, 1))
            c.decide(o, 0)
            action = c.decide(o, .1)
            if not action or action[0] == 'shift':
                continue
            direction, lease, _ = action
            # 模拟视觉线程完全不再返回、移动速度为上限且松键存在延迟。
            travel = (lease + d['release_latency_secs']) * d['max_speed'] + d['position_uncertainty']
            if direction == 'right':
                self.assertLessEqual(x + travel, 483 + 1e-6)
            else:
                self.assertGreaterEqual(x - travel, 443 - 1e-6)


class SceneTests(TestCase):
    def setUp(self):
        self.scene = RopeScene(PROFILE)

    def frame(self, monkey=True, player_x=500, player_y=522.5, offset=(0, 0)):
        frame = np.full((792, 1354, 3), 30, np.uint8)
        def put(template, cx, cy):
            h, w = template.shape[:2]
            x, y = round(cx - w / 2 + offset[0]), round(cy - h / 2 + offset[1])
            frame[y:y+h, x:x+w] = template
        put(self.scene.anchor.template, 472, 202)
        put(self.scene.player.template, player_x - self.scene.data['player_offset_x'], player_y)
        if monkey:
            put(self.scene.monkeys[0].template, 608.5, 515)
        return frame

    def test_detects_monkey_and_player(self):
        o = self.scene.observe(self.frame())
        self.assertFalse(o.reason)
        self.assertEqual(o.player.x, 500)
        self.assertIsNotNone(o.monkey)

    def test_empty_monster_region_is_idle(self):
        o = self.scene.observe(self.frame(monkey=False))
        self.assertFalse(o.reason)
        self.assertIsNone(o.monkey)

    def test_actual_monkey_animation_sequence_does_not_flicker(self):
        for i in range(8):
            frame = cv2.imread(str(Path(__file__).parent / 'fixtures' / f'archer_monkey_sequence_{i}.png'))
            o = self.scene.observe(frame)
            self.assertFalse(o.reason)
            self.assertEqual(o.monkey is not None, i >= 3)

    def test_other_floor_or_left_monkey_ignored(self):
        for x, y in ((400, 510), (700, 300)):
            f = self.frame(monkey=False)
            t = self.scene.monkeys[0].template
            f[y:y+t.shape[0], x:x+t.shape[1]] = t
            self.assertIsNone(self.scene.observe(f).monkey)

    def test_halo_survives_body_covered_with_loot(self):
        # 原始第六帧的头/身体区叠上掉落物图块，光圈区保持原像素。
        frame = cv2.imread(str(Path(__file__).parent / 'fixtures' / 'archer_halo_body_occluded.png'))
        o = self.scene.observe(frame)
        region = self.scene.region(frame, o.anchor, self.scene.data['monkey_roi'])
        self.assertTrue(all(self.scene.find_in(f, region) is None for f in self.scene.monkeys))
        self.assertFalse(o.reason)
        self.assertIsNotNone(o.monkey)
        self.assertEqual(o.monkey_source, '头顶光圈')
        self.assertAlmostEqual(o.monkey.y, 414, delta=2)
        # 光圈消失而钱币仍在时，本帧立即撤销目标，不沿用上一帧。
        frame[394:420, 555:1150] = 30
        self.assertIsNone(self.scene.observe(frame).monkey)

    def test_halo_does_not_confuse_coins_bars_or_loot(self):
        for shape in ('bar', 'line', 'coin', 'loot'):
            frame = self.frame(monkey=False)
            if shape == 'bar':
                cv2.rectangle(frame, (700, 485), (735, 490), (140, 255, 255), -1)
            elif shape == 'line':
                cv2.rectangle(frame, (700, 485), (735, 488), (140, 255, 255), -1)
            elif shape == 'coin':
                cv2.circle(frame, (715, 490), 10, (140, 255, 255), -1)
            else:
                fixture = cv2.imread(str(Path(__file__).parent / 'fixtures' / 'archer_monkey_sequence_6.png'))
                frame[480:500, 650:1000] = fixture[450:470, 640:990]
            with self.subTest(shape=shape):
                self.assertIsNone(self.scene.observe(frame).monkey)

    def test_halo_on_left_or_other_floor_is_ignored(self):
        template = cv2.imread(str(PROFILE.parent / 'monkey_halo.png'))
        for x, y in ((400, 486), (700, 380), (700, 550)):
            frame = self.frame(monkey=False)
            frame[y:y+template.shape[0], x:x+template.shape[1]] = template
            self.assertIsNone(self.scene.observe(frame).monkey)

    def test_halo_only_attack_stops_when_halo_disappears(self):
        frame = self.frame(monkey=False, player_x=463)
        template = cv2.imread(str(PROFILE.parent / 'monkey_halo.png'))
        frame[480:492, 700:742] = template
        controller = ArcherController(self.scene.data)
        o = self.scene.observe(frame)
        self.assertEqual(o.monkey_source, '头顶光圈')
        controller.decide(o, 0)
        controller.facing_right = True
        self.assertEqual(controller.decide(o, .1)[0], 'shift')
        frame[480:492, 700:742] = 30
        self.assertIsNone(controller.decide(self.scene.observe(frame), .2))

    def test_offset_window_uses_landmark(self):
        o = self.scene.observe(self.frame(offset=(15, -28)))
        self.assertFalse(o.reason)
        self.assertEqual(o.player.x - o.anchor.x, 28)
        self.assertIsNotNone(o.monkey)

    def test_dropped_player_stops(self):
        self.assertTrue(self.scene.observe(self.frame(player_y=582.5)).reason)

    def test_player_outside_safety_bounds_stops(self):
        self.assertTrue(self.scene.observe(self.frame(player_x=520)).reason)

    def test_missing_landmark_stops(self):
        self.assertTrue(self.scene.observe(np.full((792, 1354, 3), 30, np.uint8)).reason)

    def test_cached_anchor_follows_scroll_and_reacquires_large_scroll(self):
        self.scene.observe(self.frame())
        for shift in (-20, -50, 50):
            o = self.scene.observe(self.frame(offset=(0, shift)))
            self.assertFalse(o.reason)
            self.assertEqual(o.anchor.y, 202 + shift)
        self.assertTrue(self.scene.observe(np.full((792, 1354, 3), 30, np.uint8)).reason)

    def test_real_failure_preview_and_native_capture(self):
        for name, expected_x in [('archer_failed_preview.png', 480), ('archer_native_capture.png', 500)]:
            with self.subTest(name=name):
                frame = cv2.imread(str(Path(__file__).parent / 'fixtures' / name))
                self.assertIsNotNone(frame)
                o = self.scene.observe(frame)
                self.assertFalse(o.reason)
                self.assertAlmostEqual(o.player.x, expected_x, delta=2)
                self.assertGreaterEqual(o.player.score, self.scene.data['player_threshold'])

    def test_platform_failure_uses_foreground_and_preserves_geometry(self):
        frame = cv2.imread(str(Path(__file__).parent / 'fixtures' / 'archer_platform_failure.png'))
        o = self.scene.observe(frame)
        self.assertFalse(o.reason)
        self.assertEqual((o.anchor.x, o.anchor.y), (477, 129))
        self.assertGreater(o.anchor.score, .99)
        # 实际平台边界不变；新版回位目标向左移 30px。
        self.assertEqual(o.anchor.x + self.scene.data['safe_left'], 290)
        self.assertEqual(o.anchor.x + self.scene.data['safe_right'], 516)
        self.assertEqual(o.anchor.x + self.scene.data['target_x'], 468)
        self.assertIsNotNone(o.monkey)

    def test_changed_surrounding_background_does_not_move_anchor(self):
        frame = self.frame()
        rng = np.random.default_rng(42)
        for y1, y2 in [(165, 189), (215, 235)]:
            frame[y1:y2, 430:550] = rng.integers(0, 256, (y2-y1, 120, 3), dtype=np.uint8)
        o = self.scene.observe(frame)
        self.assertFalse(o.reason)
        self.assertEqual((o.anchor.x, o.anchor.y), (472, 202))

    def test_weak_platform_candidate_is_rejected(self):
        with mock.patch.object(self.scene.anchor, 'best', return_value=Hit(472, 202, .90)):
            o = self.scene.observe(self.frame())
        self.assertIsNone(o.anchor)
        self.assertIn('0.900', o.reason)

    def test_reference_template_still_supported(self):
        frame = self.frame(monkey=False)
        frame[510:535, 440:555] = 30
        finder, offset = self.scene.players[1]
        h, w = finder.template.shape[:2]
        x, y = round(500 - offset - w / 2), round(522.5 - h / 2)
        frame[y:y+h, x:x+w] = finder.template
        o = self.scene.observe(frame)
        self.assertFalse(o.reason)
        self.assertEqual(o.player.x, 500)

    def test_raw_candidate_below_threshold_is_not_accepted(self):
        with mock.patch.object(self.scene.player, 'best', return_value=Hit(30, 30, .844)), \
             mock.patch.object(self.scene.players[1][0], 'best', return_value=None), \
             mock.patch.object(self.scene.player_parts[0][0], 'find', return_value=None), \
             mock.patch.object(self.scene.player_parts[1][0], 'find', return_value=None):
            o = self.scene.observe(self.frame())
        self.assertIsNone(o.player)
        self.assertIn('0.844', o.reason)

    def occluded_frame(self, fixture_name, marker, x=500, y=522.5):
        frame = self.frame()
        frame[510:540, 440:555] = 30
        crop = cv2.imread(str(Path(__file__).parent / 'fixtures' / fixture_name))
        px, py = round(x - marker[0]), round(y - marker[1])
        frame[py:py+crop.shape[0], px:px+crop.shape[1]] = crop
        return frame

    def test_both_user_occlusion_samples_locate_same_player(self):
        samples = [('archer_name_occluded.png', (116, 111.5), '3/4'),
                   ('archer_name_occluded_close.png', (35, 132.5), '2/4')]
        for name, marker, count in samples:
            with self.subTest(name=name):
                o = self.scene.observe(self.occluded_frame(name, marker))
                self.assertFalse(o.reason)
                self.assertEqual((o.player.x, o.player.y), (500, 522.5))
                self.assertEqual(o.player_source, f'称号局部 {count}')

    def test_full_badge_is_preferred_and_recovers_after_occlusion(self):
        self.scene.observe(self.occluded_frame('archer_name_occluded.png', (116, 111.5)))
        o = self.scene.observe(self.frame())
        self.assertEqual(o.player_source, '完整称号牌')
        self.assertEqual(o.player.x, 500)

    def test_partial_badge_keeps_continuous_attack_state(self):
        controller = ArcherController(self.scene.data)
        first = self.scene.observe(self.frame(player_x=463))
        controller.decide(first, 0)
        controller.facing_right = True
        frames = [self.frame(player_x=463), self.occluded_frame('archer_name_occluded.png', (116, 111.5), x=463),
                  self.occluded_frame('archer_name_occluded_close.png', (35, 132.5), x=463), self.frame(player_x=463)]
        for i, frame in enumerate(frames):
            observation = self.scene.observe(frame)
            self.assertEqual(controller.decide(observation, .1 + i * .1)[0], 'shift')

    def test_fully_covered_badge_does_not_reuse_last_position(self):
        self.scene.observe(self.occluded_frame('archer_name_occluded.png', (116, 111.5)))
        frame = self.frame()
        frame[510:540, 440:555] = 30
        o = self.scene.observe(frame)
        self.assertIsNone(o.player)
        self.assertTrue(o.reason)

    def test_partial_recognition_still_checks_floor_and_edge(self):
        for x, y in [(520, 522.5), (500, 582.5)]:
            with self.subTest(x=x, y=y):
                o = self.scene.observe(self.occluded_frame('archer_name_occluded_close.png', (35, 132.5), x, y))
                self.assertTrue(o.reason)
                self.assertIsNotNone(o.player)


class PartialBadgeTests(TestCase):
    def setUp(self):
        self.scene = RopeScene(PROFILE)
        self.finder = self.scene.player_parts[0][0]
        self.template = self.scene.player.template

    def image(self, indices, origin=(20, 20)):
        frame = np.full((80, 250, 3), 30, np.uint8)
        for i in indices:
            left, part = self.finder.parts[i]
            x, y = origin[0] + left, origin[1]
            frame[y:y+part.shape[0], x:x+part.shape[1]] = part
        return frame

    def test_only_one_visible_fragment_is_insufficient(self):
        for index in range(4):
            self.assertIsNone(self.finder.find(self.image([index])))

    def test_either_half_can_locate_full_badge_center(self):
        for indices in ([0, 1], [2, 3], [0, 3]):
            result = self.finder.find(self.image(indices))
            self.assertIsNotNone(result)
            hit, count = result
            self.assertEqual(count, 2)
            self.assertEqual((hit.x, hit.y), (61, 26.5))

    def test_fragments_from_two_different_people_are_not_combined(self):
        frame = self.image([0], (10, 20))
        other = self.image([1], (130, 20))
        frame[:, 130:] = other[:, 130:]
        self.assertIsNone(self.finder.find(frame))

    def test_two_consistent_but_separate_badges_are_ambiguous(self):
        frame = self.image([0, 1], (10, 20))
        other = self.image([2, 3], (130, 20))
        frame[:, 130:] = other[:, 130:]
        self.assertIsNone(self.finder.find(frame))

    def test_vision_disabled_fails_without_input(self):
        bot = mock.Mock()
        with self.assertRaises(ValueError):
            run(bot, Config(vision_enabled=False))
        bot.try_tap.assert_not_called()

    def test_missing_template_fails_without_input(self):
        bot = mock.Mock()
        with self.assertRaises(FileNotFoundError):
            run(bot, Config(archer_profile='missing_profile.json'))
        bot.try_tap.assert_not_called()


class InputTests(TestCase):
    def bot(self):
        bot = Bot.__new__(Bot)
        bot.poll_hotkeys = mock.Mock()
        bot.foreground = mock.Mock(return_value=True)
        bot.paused = bot.quitting = False
        bot.down, bot.up = mock.Mock(), mock.Mock()
        return bot

    def test_lost_focus_does_not_send_stale_action(self):
        b = self.bot()
        b.foreground.return_value = False
        self.assertFalse(b.try_tap('shift', .055))
        b.down.assert_not_called()

    def test_pause_does_not_send_stale_action(self):
        b = self.bot()
        b.paused = True
        self.assertFalse(b.try_tap('right', .02))
        b.down.assert_not_called()

    def test_focus_loss_releases_held_key(self):
        b = self.bot()
        b.foreground.side_effect = [True, False]
        self.assertFalse(b.try_tap('right', .035))
        b.up.assert_called_once_with('right')

    def test_stop_during_press_releases_key(self):
        b = self.bot()
        b.poll_hotkeys.side_effect = [None, BotStopped()]
        with self.assertRaises(BotStopped):
            b.try_tap('shift', .055)
        b.up.assert_called_once_with('shift')


class FailureSnapshotTests(TestCase):
    def test_keeps_latest_image_per_error_category_with_rate_limit(self):
        with tempfile.TemporaryDirectory() as folder:
            saver = FailureSnapshots(Path(folder))
            frame = np.full((20, 20, 3), 30, np.uint8)
            failure = Observation(reason='未认到人物名字牌')
            path = saver.record(frame, failure, 0)
            self.assertTrue(path.exists())
            self.assertIsNone(saver.record(frame, failure, 1))
            self.assertEqual(saver.record(frame, failure, 6), path)
            self.assertEqual(len(list(Path(folder).iterdir())), 2)
            metadata = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
            self.assertEqual(metadata['version'], ARCHER_VERSION)
            self.assertEqual(metadata['reason'], failure.reason)

    def test_valid_observation_creates_no_diagnostic_file(self):
        with tempfile.TemporaryDirectory() as folder:
            saver = FailureSnapshots(Path(folder))
            self.assertIsNone(saver.record(np.zeros((20, 20, 3), np.uint8), Observation(), 0))
            self.assertFalse(list(Path(folder).iterdir()))


class CaptureSafetyTests(TestCase):
    def check_frame(self, duration, action):
        bot, scene, controller, inputs = mock.Mock(), mock.Mock(), mock.Mock(), mock.Mock()
        scene.data = RopeScene(PROFILE).data
        scene.observe.return_value = Observation(Hit(400, 522.5, 1), Hit(472, 202, 1))
        controller.decide.return_value = action
        controller.moving = None  # 转向持键也必须算入截图期间的时间。
        inputs.__enter__ = mock.Mock(return_value=inputs)
        inputs.__exit__ = mock.Mock(return_value=False)
        inputs.epoch, inputs.key = 0, 'right'
        bot.gate.return_value = False
        bot.wait.side_effect = BotStopped()
        with mock.patch('autofarm.rope_archer.RopeScene', return_value=scene), \
             mock.patch('autofarm.rope_archer.ArcherController', return_value=controller), \
             mock.patch('autofarm.held_input.HeldInput', return_value=inputs), \
             mock.patch('autofarm.rope_archer.V.capture'), \
             mock.patch('autofarm.rope_archer.time.monotonic', side_effect=[10, 10 + duration]):
            with self.assertRaises(BotStopped):
                run(bot, Config())
        return controller, inputs

    def test_slow_frame_releases_and_does_not_act(self):
        controller, inputs = self.check_frame(.13, ('right', .1, 'move'))
        inputs.clear.assert_called()
        inputs.apply.assert_not_called()
        controller.decide.assert_not_called()

    def test_held_direction_deadline_includes_capture_time_even_when_turning(self):
        _, inputs = self.check_frame(.08, ('right', .1, 'turn'))
        inputs.apply.assert_called_once_with('right', 10.1, 0)
