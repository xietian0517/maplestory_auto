"""Offline input events: v2.2.6 behavior, bounded jitter, counts and cleanup."""
from dataclasses import replace
import random
import queue
from unittest import TestCase, mock

from autofarm import plans
from autofarm.bot import Bot, BotStopped
from autofarm.held_input import HeldInput
from autofarm.buffs import BuffSlot
from farm import Config


class SweepJumpTests(TestCase):
    def setUp(self):
        self.now = 0.0
        self.on_sleep = lambda: None
        self.events, self.timeline = [], []
        self.bot = Bot.__new__(Bot)
        self.bot.api = mock.Mock()
        self.bot.hwnd = 1
        self.bot.paused = self.bot.quitting = self.bot._fg_lost = False
        self.bot._prev = {'f11': False, 'f12': False}
        self.bot.held = set()
        self.bot.buff_commands = queue.Queue()
        self.bot.log = mock.Mock()
        self.bot.api.async_pressed.return_value = False
        self.bot.api.get_foreground.return_value = 1
        self.bot.api.is_iconic.return_value = False
        self.bot.api.send_key.side_effect = self.key_event
        for patch in (mock.patch('autofarm.plans.time.monotonic', side_effect=lambda: self.now),
                      mock.patch('autofarm.plans.time.sleep', side_effect=self.sleep),
                      mock.patch.object(HeldInput, '_watch', return_value=None)):
            patch.start()
            self.addCleanup(patch.stop)

    def sleep(self, seconds):
        self.now += seconds
        self.on_sleep()

    def key_event(self, key, up):
        self.events.append((key, up, frozenset(self.bot.held)))
        self.timeline.append((key, up, self.now))

    def config(self, **kwargs):
        return replace(Config(plan='sweep_jump', sweep_right_attacks=3, sweep_left_attacks=2,
                              sweep_noise_ratio=0, buff_slots=()), **kwargs)

    def stop_after_attacks(self, total):
        completed = 0
        def log(message):
            nonlocal completed
            if message.startswith('[第'):
                completed += 1
                if completed == total:
                    self.bot.quitting = True
        self.bot.log.side_effect = log

    def test_zero_noise_matches_v226_timing_and_unbroken_direction(self):
        self.stop_after_attacks(10)
        with self.assertRaises(BotStopped):
            plans.sweep_jump(self.bot, self.config())
        directions = [(k, up) for k, up, _ in self.events if k in plans.SIDES]
        self.assertEqual(directions, [('right', False), ('right', True), ('left', False), ('left', True)] * 2)
        for key in ('alt', 'shift'):
            pressed_sides = [next(iter(held & set(plans.SIDES)))
                             for k, up, held in self.events if k == key and not up]
            self.assertEqual(pressed_sides, (['right'] * 3 + ['left'] * 2) * 2)
        self.assertAlmostEqual(self.now, 8)
        jump_starts = [t for k, up, t in self.timeline if k == 'alt' and not up]
        attack_starts = [t for k, up, t in self.timeline if k == 'shift' and not up]
        for jump, attack in zip(jump_starts, attack_starts):
            self.assertAlmostEqual(attack - jump, .2)
        self.assertFalse(self.bot.held)

    def test_noise_changes_each_timing_and_preserves_ten_each_way(self):
        self.stop_after_attacks(40)
        rng = random.Random(123)
        with mock.patch('autofarm.plans.random.uniform', side_effect=rng.uniform), \
             self.assertRaises(BotStopped):
            plans.sweep_jump(self.bot, self.config(sweep_right_attacks=10, sweep_left_attacks=10, sweep_noise_ratio=.15))
        expected = (['right'] * 10 + ['left'] * 10) * 2
        for key in ('alt', 'shift'):
            sides = [next(iter(held & set(plans.SIDES))) for k, up, held in self.events if k == key and not up]
            self.assertEqual(sides, expected)
        starts = [t for k, up, t in self.timeline if k == 'alt' and not up]
        cycles = [b - a for a, b in zip(starts, starts[1:])]
        self.assertGreater(len(set(round(v, 5) for v in cycles)), 30)
        self.assertTrue(all(.68 - 1e-8 <= v <= .92 + 1e-8 for v in cycles))
        direction_events = [(k, up, t) for k, up, t in self.timeline if k in plans.SIDES]
        self.assertEqual(len(direction_events), 8)
        for release, press in zip(direction_events[1::2], direction_events[2::2]):
            self.assertEqual(release[2], press[2])

    def test_samples_respect_bounds_and_recovery_floor(self):
        rng = random.Random(42)
        cfg = self.config(sweep_noise_ratio=.3)
        bounds = plans.sweep_timing_bounds(cfg)
        with mock.patch('autofarm.plans.random.uniform', side_effect=rng.uniform):
            samples = [plans.random_sweep_timing(cfg) for _ in range(200)]
        for sample in samples:
            for value, (lo, hi) in zip(sample, bounds):
                self.assertTrue(lo <= value <= hi)
            self.assertGreaterEqual(sample[3], sum(sample[:3]) + .1)
        for column in zip(*samples):
            self.assertGreater(len(set(column)), 100)
        cfg = self.config(sweep_noise_ratio=.3, sweep_jump_hold_secs=.25,
                          sweep_jump_rise_secs=.3, sweep_attack_hold_secs=.25, sweep_cycle_secs=1)
        with mock.patch('autofarm.plans.random.uniform', side_effect=[.25, .3, .25, .7]):
            self.assertAlmostEqual(plans.random_sweep_timing(cfg)[3], .9)

    def test_pause_and_focus_release_then_resume_remaining_movement(self):
        for reason in ('pause', 'focus'):
            with self.subTest(reason=reason):
                self.now = 0
                self.events.clear()
                def interruption():
                    blocked = .04 <= self.now < .1
                    self.bot.paused = blocked if reason == 'pause' else False
                    self.bot.api.get_foreground.return_value = 2 if blocked and reason == 'focus' else 1
                self.on_sleep = interruption
                with HeldInput(self.bot) as inputs:
                    interrupted = plans._SweepMotion(self.bot, inputs, 'right').wait(.12)
                self.assertTrue(interrupted)
                self.assertGreaterEqual(self.now, .18)
                self.assertEqual([(k, up) for k, up, _ in self.events],
                                 [('right', False), ('right', True), ('right', False), ('right', True)])
                self.assertFalse(self.bot.held)

    def test_focus_loss_during_jump_restarts_jump_without_ground_attack(self):
        with HeldInput(self.bot) as inputs:
            motion = plans._SweepMotion(self.bot, inputs, 'right')
            self.on_sleep = lambda: setattr(self.bot.api.get_foreground, 'return_value',
                                           2 if .005 <= self.now < .15 else 1)
            self.assertFalse(motion.jump_attack(self.config()))
            self.assertFalse(any(k == 'shift' for k, _, _ in self.events))
            self.on_sleep = lambda: None
            self.bot.api.get_foreground.return_value = 1
            self.assertTrue(motion.jump_attack(self.config()))
        self.assertEqual(sum(k == 'alt' and not up for k, up, _ in self.events), 2)
        self.assertEqual(sum(k == 'shift' and not up for k, up, _ in self.events), 1)

    def test_stop_during_attack_releases_attack_and_direction(self):
        def stop():
            if 'shift' in self.bot.held:
                self.bot.quitting = True
        self.on_sleep = stop
        with self.assertRaises(BotStopped), HeldInput(self.bot) as inputs:
            plans._SweepMotion(self.bot, inputs, 'right').jump_attack(self.config())
        self.assertFalse(self.bot.held)

    def test_buff_off_has_no_screenshots_potion_or_extra_wait(self):
        self.stop_after_attacks(10)
        cfg = self.config(potion_key='end', potion_every=1, buff_key='home',
                          buff_slots=(), name_template='',
                          move_secs=(20, 20), attack_gap_secs=(20, 20), switch_gap_secs=(20, 20))
        with mock.patch('autofarm.plans.V.capture', side_effect=AssertionError('screenshot')), \
             mock.patch('autofarm.plans.maybe_buff', side_effect=AssertionError('Buff')), \
             mock.patch('autofarm.plans.B.drink_potion', side_effect=AssertionError('potion')), \
             mock.patch('autofarm.plans.B.switch_side', side_effect=AssertionError('side wait')):
            with self.assertRaises(BotStopped):
                plans.sweep_jump(self.bot, cfg)
        self.assertTrue(all(k in ('left', 'right', 'alt', 'shift') for k, _, _ in self.events))
        self.assertAlmostEqual(self.now, 8)

    def test_immediate_and_timed_buffs_release_keys_and_preserve_counts(self):
        self.stop_after_attacks(20)
        slots = (BuffSlot('home', (2, 2), (.3, .3)), BuffSlot('ins', (4, 4), (.3, .3)))
        with self.assertRaises(BotStopped):
            plans.sweep_jump(self.bot, self.config(buff_slots=slots, buff_start_immediately=True,
                                                  buff_hold_secs=.05))
        for key in ('alt', 'shift'):
            sides = [next(iter(held & set(plans.SIDES))) for k, up, held in self.events if k == key and not up]
            self.assertEqual(sides, (['right'] * 3 + ['left'] * 2) * 4)
        self.assertEqual([k for k, up, _ in self.events if not up][:2], ['home', 'ins'])
        for key, interval in [('home', 2), ('ins', 4)]:
            starts = [t for k, up, t in self.timeline if k == key and not up]
            self.assertGreater(len(starts), 2)
            for a, b in zip(starts, starts[1:]):
                self.assertGreaterEqual(b - a, interval)
            for k, up, held in self.events:
                if k == key and not up:
                    self.assertFalse(held & {'left', 'right', 'alt', 'shift'})
        for i, (key, up, at) in enumerate(self.timeline):
            if key in ('home', 'ins') and not up:
                releases = [t for k, released, t in self.timeline[:i] if k in plans.SIDES and released]
                self.assertGreaterEqual(at - (releases[-1] if releases else 0) + 1e-8, .15)
        self.assertFalse(self.bot.held)

    def test_live_enable_force_and_disable_between_groups(self):
        slot = BuffSlot('home', (100, 100), (0, 0))
        completed = 0
        def log(message):
            nonlocal completed
            if not message.startswith('[第'):
                return
            completed += 1
            if completed == 1:
                self.bot.buff_commands.put(('hold', .05))
                self.bot.buff_commands.put(('replace', (slot,)))
            elif completed == 2:
                self.bot.buff_commands.put(('now', None))
            elif completed == 3:
                self.bot.buff_commands.put(('now', None))
                self.bot.buff_commands.put(('replace', ()))
            elif completed == 5:
                self.bot.quitting = True
        self.bot.log.side_effect = log
        with self.assertRaises(BotStopped):
            plans.sweep_jump(self.bot, self.config())
        self.assertEqual(sum(k == 'home' and not up for k, up, _ in self.events), 2)
        self.assertEqual(sum(k == 'shift' and not up for k, up, _ in self.events), 5)
        sides = [next(iter(held & set(plans.SIDES))) for k, up, held in self.events if k == 'shift' and not up]
        self.assertEqual(sides, ['right'] * 3 + ['left'] * 2)
        self.assertIn('关闭', self.bot.buff_status)

    def test_focus_loss_during_buff_retries_without_advancing_timer(self):
        self.stop_after_attacks(2)
        slot = BuffSlot('home', (100, 100), (0, 0))
        def lose_focus():
            self.bot.api.get_foreground.return_value = 2 if .16 <= self.now < .3 else 1
        self.on_sleep = lose_focus
        with self.assertRaises(BotStopped):
            plans.sweep_jump(self.bot, self.config(buff_slots=(slot,), buff_start_immediately=True,
                                                  buff_hold_secs=.05))
        home_starts = [t for k, up, t in self.timeline if k == 'home' and not up]
        self.assertEqual(len(home_starts), 2)
        self.assertGreaterEqual(home_starts[1], .3)
        sent = [call.args[0] for call in self.bot.log.call_args_list if call.args[0].startswith('[Buff] 已发送')]
        self.assertEqual(len(sent), 1)
        self.assertFalse(self.bot.held)

    def test_invalid_noise_and_counts_fail_before_keys(self):
        for value in (-.01, .31, float('nan'), float('inf')):
            with self.subTest(noise=value), self.assertRaises(ValueError):
                plans.sweep_jump(self.bot, self.config(sweep_noise_ratio=value))
        for value in (0, -1, 1001, 1.5, True):
            with self.subTest(count=value), self.assertRaises(ValueError):
                plans.sweep_jump(self.bot, self.config(sweep_right_attacks=value))
        self.assertFalse(self.events)
