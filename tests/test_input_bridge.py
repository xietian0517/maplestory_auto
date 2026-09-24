import threading
import time
import unittest

from autofarm.input_bridge import InputController, validate_action


def action(**updates):
    result = dict(id='test-request-1', issued_at=time.time(), hwnd=123, keys=['shift'], ms=50)
    result.update(updates)
    return result


class FakeAdapter:
    def __init__(self):
        self.foreground = True
        self.emergency = False
        self.events = []
        self.invalid = False
        self.fail_down = False

    def validate_target(self, hwnd):
        if self.invalid:
            raise ValueError('wrong process')

    def is_target_foreground(self, hwnd):
        return self.foreground

    def emergency_pressed(self):
        return self.emergency

    def send_key(self, key, up):
        self.events.append((key, up))
        if self.fail_down and not up:
            raise OSError('injection failed')


class BridgeTests(unittest.TestCase):
    def test_invalid_and_stale_actions(self):
        for changes in [dict(keys=['enter']), dict(keys=['shift', 'shift']), dict(keys=['left', 'right']),
                        dict(ms=2001), dict(ms=True), dict(hwnd=True), dict(issued_at=time.time()-10),
                        dict(issued_at=float('nan')), dict(command='shell'),
                        dict(keys=['alt', 'esc']), dict(keys=['ctrl', 'esc']), dict(hwnd=2**64)]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_action(action(**changes))

    def test_background_and_wrong_process_never_receive_input(self):
        for attr in ['foreground', 'invalid']:
            adapter = FakeAdapter()
            setattr(adapter, attr, attr == 'invalid')
            with self.assertRaises(ValueError):
                InputController(adapter).execute(action())
            self.assertEqual(adapter.events, [])

    def test_normal_action_releases_and_duplicate_cannot_replay(self):
        adapter = FakeAdapter()
        controller = InputController(adapter)
        request = action()
        result = controller.execute(request)
        self.assertFalse(result['interrupted'])
        self.assertEqual(adapter.events, [('shift', False), ('shift', True)])
        with self.assertRaises(ValueError):
            controller.execute(request)
        self.assertEqual(len(adapter.events), 2)

    def test_focus_loss_and_emergency_interrupt_hold(self):
        for attr in ['foreground', 'emergency']:
            adapter = FakeAdapter()
            controller = InputController(adapter)
            timer = threading.Timer(.04, lambda: setattr(adapter, attr, attr == 'emergency'))
            timer.start()
            try:
                result = controller.execute(action(ms=800))
            finally:
                timer.join()
            self.assertTrue(result['interrupted'])
            self.assertLess(result['elapsed_ms'], 500)
            self.assertEqual(adapter.events[-1], ('shift', True))
            self.assertFalse(controller.held)

    def test_failed_keydown_still_attempts_release(self):
        adapter = FakeAdapter()
        adapter.fail_down = True
        controller = InputController(adapter)
        with self.assertRaises(OSError):
            controller.execute(action())
        self.assertEqual(adapter.events, [('shift', False), ('shift', True)])

    def test_stop_rejects_future_actions(self):
        adapter = FakeAdapter()
        controller = InputController(adapter)
        controller.stop()
        with self.assertRaises(ValueError):
            controller.execute(action())
        self.assertEqual(adapter.events, [])

    def test_concurrent_request_cannot_release_running_action(self):
        adapter = FakeAdapter()
        controller = InputController(adapter)
        worker = threading.Thread(target=lambda: controller.execute(action(ms=150)))
        worker.start()
        try:
            deadline = time.monotonic() + 1
            while not adapter.events and time.monotonic() < deadline:
                time.sleep(.001)
            with self.assertRaises(ValueError):
                controller.execute(action(id='other-request'))
            self.assertEqual(adapter.events, [('shift', False)])
        finally:
            worker.join()
        self.assertEqual(adapter.events[-1], ('shift', True))

    def test_stop_releases_active_hold(self):
        adapter = FakeAdapter()
        controller = InputController(adapter)
        timer = threading.Timer(.04, controller.stop)
        timer.start()
        try:
            result = controller.execute(action(ms=800))
        finally:
            timer.join()
        self.assertTrue(result['interrupted'])
        self.assertLess(result['elapsed_ms'], 500)
        self.assertEqual(adapter.events, [('shift', False), ('shift', True)])

    def test_independent_watchdog_releases_expired_hold(self):
        adapter = FakeAdapter()
        controller = InputController(adapter)
        controller.held.add('right')
        controller.target = 123
        controller.deadline = time.monotonic() - 1
        controller.watch()
        self.assertEqual(adapter.events, [('right', True)])
        self.assertFalse(controller.held)
        self.assertFalse(controller.interrupted)

    def test_release_failure_stops_further_input(self):
        class ReleaseFailure(FakeAdapter):
            def send_key(self, key, up):
                super().send_key(key, up)
                if up:
                    raise OSError('cannot release')
        controller = InputController(ReleaseFailure())
        with self.assertRaises(RuntimeError):
            controller.execute(action())
        self.assertTrue(controller.stopped)


if __name__ == '__main__':
    unittest.main()
