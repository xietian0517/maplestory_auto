"""Hidden Tk window tests; fake input only, no elevation or game access."""
import time
import tkinter as tk
import unittest

from autofarm.bridge_service import BridgeService
from game_input_bridge import BridgeWindow


class FakeGame:
    def __init__(self):
        self.events = []
        self.foreground = False

    def find_game(self):
        return 123

    def validate_target(self, hwnd):
        if hwnd != 123:
            raise ValueError('wrong game')

    def is_target_foreground(self, hwnd):
        return self.foreground

    def emergency_pressed(self):
        return False

    def send_key(self, key, up):
        self.events.append((key, up))


class BridgeGuiTests(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.adapter = FakeGame()
        self.window = BridgeWindow(self.root, self.adapter,
                                   service_factory=lambda adapter, **kwargs:
                                   BridgeService(adapter, state_file=None, **kwargs))
        self.root.update_idletasks()

    def tearDown(self):
        self.window.close()

    def pump(self, seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.root.update()
            time.sleep(.01)

    def test_starting_window_does_not_press_keys(self):
        self.pump(.1)
        self.assertEqual(self.adapter.events, [])
        self.assertFalse(self.window.service.controller.stopped)
        self.assertTrue(all(button.winfo_reqwidth() > 40 for button in self.window.test_buttons))

    def test_cancel_countdown_sends_nothing(self):
        self.adapter.foreground = True
        self.window.schedule_test(['right'], 200)
        self.assertIsNotNone(self.window.pending)
        self.window.stop()
        self.pump(3.15)
        self.assertEqual(self.adapter.events, [])

    def test_manual_test_requires_game_foreground(self):
        self.window.schedule_test(['shift'], 100)
        self.pump(3.3)
        self.assertEqual(self.adapter.events, [])
        self.assertIn('未完成测试', self.window.log_widget.get('1.0', 'end'))

    def test_countdown_executes_once_after_switching_to_game(self):
        self.window.schedule_test(['alt'], 100)
        self.assertEqual(self.adapter.events, [])
        self.adapter.foreground = True
        self.pump(3.35)
        self.assertEqual(self.adapter.events, [('alt', False), ('alt', True)])


if __name__ == '__main__':
    unittest.main()
