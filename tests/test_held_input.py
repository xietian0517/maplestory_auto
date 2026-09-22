"""持续持键事件与松键监护：全部使用假 Bot，不向游戏发键。"""
from unittest import TestCase, mock

from autofarm.held_input import HeldInput
from autofarm.bot import BotStopped


class HeldInputTests(TestCase):
    def setUp(self):
        self.bot = mock.Mock(paused=False, quitting=False)
        self.bot.foreground.return_value = True
        self.inputs = HeldInput(self.bot)
        self.clock = mock.patch('autofarm.held_input.time.monotonic', return_value=0)
        self.now = self.clock.start()
        self.addCleanup(self.clock.stop)

    def test_attack_renews_without_repeated_key_events(self):
        for t in (0, .1, .2, .3, .4, .5):
            self.now.return_value = t
            self.assertTrue(self.inputs.apply('shift', t + .4, 0))
            self.inputs.check()
        self.bot.down.assert_called_once_with('shift')
        self.bot.up.assert_not_called()
        self.inputs.clear()  # 下一帧无猴子
        self.bot.up.assert_called_once_with('shift')

    def test_movement_is_one_hold_across_frames(self):
        self.inputs.apply('right', .3, 0)
        self.now.return_value = .1
        self.inputs.apply('right', .35, 0)
        self.now.return_value = .2
        self.inputs.apply('right', .4, 0)
        self.bot.down.assert_called_once_with('right')
        self.bot.up.assert_not_called()
        self.inputs.clear()
        self.bot.up.assert_called_once_with('right')

    def test_change_direction_releases_previous_key_first(self):
        self.inputs.apply('left', .4, 0)
        self.inputs.apply('right', .4, 0)
        calls = [c for c in self.bot.method_calls if c[0] in ('up', 'down')]
        self.assertEqual(calls, [mock.call.down('left'), mock.call.up('left'), mock.call.down('right')])

    def test_slow_or_stuck_capture_expires_key_without_new_frame(self):
        self.inputs.apply('right', .075, 0)
        self.now.return_value = .08
        self.inputs.check()
        self.bot.up.assert_called_once_with('right')
        self.assertIsNone(self.inputs.key)

    def test_pause_focus_loss_and_stop_release_hold(self):
        for kind in ('pause', 'focus', 'stop'):
            with self.subTest(kind=kind):
                self.bot.paused = self.bot.quitting = False
                self.bot.foreground.return_value = True
                h = HeldInput(self.bot)
                h.apply('shift', .4, 0)
                if kind == 'pause': self.bot.paused = True
                if kind == 'focus': self.bot.foreground.return_value = False
                if kind == 'stop': self.bot.quitting = True
                h.check()
                self.assertIsNone(h.key)
                self.assertEqual(h.epoch, 1)
                self.bot.up.assert_called_with('shift')

    def test_returning_focus_does_not_accept_pre_switch_frame(self):
        self.inputs.apply('shift', .4, 0)
        self.bot.foreground.return_value = False
        self.inputs.check()
        self.bot.foreground.return_value = True
        self.inputs.check()
        self.assertFalse(self.inputs.apply('shift', .4, 0))
        self.bot.down.assert_called_once_with('shift')

    def test_expired_movement_deadline_cannot_start_a_new_press(self):
        self.now.return_value = .1
        self.assertFalse(self.inputs.apply('right', .08, 0))
        self.bot.down.assert_not_called()

    def test_exception_cleanup_releases_key(self):
        with self.assertRaises(RuntimeError):
            with self.inputs:
                self.inputs.apply('shift', .4, 0)
                raise RuntimeError('capture failed')
        self.bot.up.assert_called_once_with('shift')

    def test_stop_before_apply_never_presses_key(self):
        self.bot.quitting = True
        with self.assertRaises(BotStopped):
            self.inputs.apply('shift', .4, 0)
        self.bot.down.assert_not_called()
