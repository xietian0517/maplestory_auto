import http.client
import json
from pathlib import Path
import tempfile
import time
import unittest

from autofarm.bridge_service import BridgeService, make_action, request


class FakeGame:
    def __init__(self):
        self.events = []
        self.foreground = True
        self.emergency = False

    def validate_target(self, hwnd):
        if hwnd != 123:
            raise ValueError('Wrong target')

    def is_target_foreground(self, hwnd):
        return hwnd == 123 and self.foreground

    def emergency_pressed(self):
        return self.emergency

    def send_key(self, key, up):
        self.events.append((key, up))


class BridgeProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.state = Path(self.temp.name) / 'session.json'
        self.adapter = FakeGame()
        self.service = BridgeService(self.adapter, state_file=self.state).start()

    def tearDown(self):
        self.service.close()
        self.temp.cleanup()

    def raw(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection('127.0.0.1', self.service.server.server_port, timeout=2)
        try:
            conn.request(method, path, body=body, headers=headers or {})
            result = conn.getresponse()
            return result.status, json.loads(result.read())
        finally:
            conn.close()

    def headers(self, **extra):
        return {'Authorization': 'Bearer ' + self.service.token,
                'Content-Type': 'application/json', **extra}

    def test_client_roundtrip_and_cleanup(self):
        self.assertFalse(request('health', state_file=self.state)['stopped'])
        result = request('action', make_action(123, ['right'], 30), state_file=self.state)
        self.assertTrue(result['ok'])
        self.assertEqual(self.adapter.events, [('right', False), ('right', True)])
        self.assertTrue(request('stop', state_file=self.state)['ok'])
        self.assertTrue(self.service.closed.wait(2))
        self.assertFalse(self.state.exists())

    def test_missing_wrong_token_and_browser_origin_are_rejected(self):
        for headers in [{}, {'Authorization': 'Bearer wrong'}, self.headers(Origin='https://example.com')]:
            with self.subTest(headers=list(headers)):
                status, _ = self.raw('POST', '/action', json.dumps(make_action(123, ['shift'], 30)), headers)
                self.assertEqual(status, 403)
        self.assertEqual(self.adapter.events, [])

    def test_invalid_http_bodies_never_send_keys(self):
        for body, headers in [('not json', self.headers()), ('x'*1025, self.headers()),
                              ('{}', self.headers(**{'Content-Type': 'text/plain'}))]:
            status, result = self.raw('POST', '/action', body, headers)
            self.assertEqual(status, 409)
            self.assertFalse(result['ok'])
        self.assertEqual(self.adapter.events, [])

    def test_wrong_target_stale_and_duplicate_requests(self):
        invalid = make_action(999, ['shift'], 30)
        stale = make_action(123, ['shift'], 30)
        stale['issued_at'] -= 10
        for data in [invalid, stale]:
            with self.assertRaises(RuntimeError):
                request('action', data, state_file=self.state)
        self.assertEqual(self.adapter.events, [])
        data = make_action(123, ['shift'], 30)
        request('action', data, state_file=self.state)
        with self.assertRaises(RuntimeError):
            request('action', data, state_file=self.state)
        self.assertEqual(len(self.adapter.events), 2)

    def test_only_named_endpoints_are_available(self):
        for endpoint in ['/shell', '/focus', '/type', '/launch']:
            self.assertEqual(self.raw('POST', endpoint, '{}', self.headers())[0], 404)

    def test_session_and_idle_expiry(self):
        self.service.close()
        for limits in [dict(max_seconds=.08, idle_seconds=10), dict(max_seconds=10, idle_seconds=.08)]:
            self.service = BridgeService(self.adapter, state_file=self.state, **limits).start()
            self.assertTrue(self.service.closed.wait(2))
            self.assertFalse(self.state.exists())
            self.assertTrue(self.service.controller.stopped)

    def test_health_poll_does_not_extend_idle_session(self):
        self.service.last_action = time.monotonic() - 290
        previous = self.service.last_action
        request('health', state_file=self.state)
        self.assertEqual(previous, self.service.last_action)

    def test_emergency_while_idle_shuts_down(self):
        self.adapter.emergency = True
        self.assertTrue(self.service.closed.wait(2))
        self.assertFalse(self.state.exists())

    def test_old_session_does_not_delete_new_session_file(self):
        self.state.write_text(json.dumps({'pid': 0, 'token': 'new-session'}), encoding='utf-8')
        self.service.close()
        self.assertEqual(json.loads(self.state.read_text())['token'], 'new-session')


if __name__ == '__main__':
    unittest.main()
