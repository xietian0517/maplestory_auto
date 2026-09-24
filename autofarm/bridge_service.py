"""Local authenticated transport for bounded game input. No Windows UI code here."""
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import threading
import time
import urllib.error
import urllib.request
import uuid

from .input_bridge import InputController

STATE_FILE = Path(os.environ.get('LOCALAPPDATA', str(Path.home()))) / 'MapleInputBridge' / 'session.json'


def make_action(hwnd, keys, ms):
    return dict(id=uuid.uuid4().hex, issued_at=time.time(), hwnd=hwnd, keys=keys, ms=ms)


class BridgeService:
    def __init__(self, adapter, state_file=STATE_FILE, max_seconds=900, idle_seconds=300, log=None):
        self.adapter = adapter
        self.controller = InputController(adapter)
        self.state_file = Path(state_file) if state_file is not None else None
        self.max_seconds = max_seconds
        self.idle_seconds = idle_seconds
        self.log = log or (lambda message: None)
        self.token = secrets.token_urlsafe(32)
        self.started = time.monotonic()
        self.last_action = self.started
        self.done = threading.Event()
        self.closed = threading.Event()
        self.server = None
        self.threads = []

    def health(self):
        return dict(ok=True, pid=os.getpid(), stopped=self.controller.stopped,
                    remaining_seconds=max(0, round(self.max_seconds - (time.monotonic() - self.started))))

    def execute(self, data):
        result = self.controller.execute(data)
        self.last_action = time.monotonic()
        self.log('{} · {} ms{}'.format(' + '.join(data['keys']), result['elapsed_ms'],
                                      ' · 提前松键' if result['interrupted'] else ' · 已松键'))
        return result

    def stop(self):
        errors = self.controller.stop()
        self.done.set()
        return errors

    def start(self):
        service = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def setup(self):
                super().setup()
                self.connection.settimeout(3)

            def respond(self, code, value):
                raw = json.dumps(value).encode()
                self.send_response(code)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(raw)))
                self.end_headers()
                try:
                    self.wfile.write(raw)
                except OSError:
                    pass  # Input outcome is unknown to a disconnected client; no retry.

            def authorized(self):
                candidate = self.headers.get('Authorization', '').encode('utf-8')
                expected = ('Bearer ' + service.token).encode('ascii')
                return not self.headers.get('Origin') and hmac.compare_digest(candidate, expected)

            def do_GET(self):
                if not self.authorized():
                    return self.respond(403, {'error': 'Unauthorized'})
                if self.path != '/health':
                    return self.respond(404, {'error': 'Unknown endpoint'})
                return self.respond(200, service.health())

            def do_POST(self):
                if not self.authorized():
                    return self.respond(403, {'error': 'Unauthorized'})
                if self.path == '/stop':
                    errors = service.stop()
                    return self.respond(200, {'ok': not errors, 'release_errors': errors})
                if self.path != '/action':
                    return self.respond(404, {'error': 'Unknown endpoint'})
                try:
                    if self.headers.get('Content-Type') != 'application/json' or self.headers.get('Transfer-Encoding'):
                        raise ValueError('Plain JSON with Content-Length required')
                    size = int(self.headers.get('Content-Length', '0'))
                    if not 0 < size <= 1024:
                        raise ValueError('Invalid body size')
                    data = json.loads(self.rfile.read(size))
                    return self.respond(200, service.execute(data))
                except (ValueError, RuntimeError, OSError) as exc:
                    return self.respond(409, {'ok': False, 'error': str(exc)})
                except Exception:
                    service.stop()
                    return self.respond(500, {'ok': False, 'error': 'Unexpected error; bridge stopped'})

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.daemon_threads = True
        self.server.block_on_close = False
        self.started = self.last_action = time.monotonic()
        try:
            if self.state_file is not None:
                self.state_file.parent.mkdir(parents=True, exist_ok=True)
                state = dict(port=self.server.server_port, token=self.token, pid=os.getpid(),
                             started_at=time.time())
                temporary = self.state_file.with_name('session-' + uuid.uuid4().hex + '.tmp')
                try:
                    temporary.write_text(json.dumps(state), encoding='utf-8')
                    temporary.replace(self.state_file)
                finally:
                    if temporary.exists():
                        temporary.unlink()
            server_thread = threading.Thread(target=self.server.serve_forever,
                                             kwargs={'poll_interval': .05}, daemon=True)
            monitor_thread = threading.Thread(target=self._monitor, daemon=True)
            self.threads = [server_thread, monitor_thread]
            server_thread.start()
            monitor_thread.start()
        except Exception:
            self.server.server_close()
            self._cleanup_state()
            raise
        return self

    def _cleanup_state(self):
        if self.state_file is None:
            return
        try:
            state = json.loads(self.state_file.read_text(encoding='utf-8'))
            if state.get('pid') == os.getpid() and state.get('token') == self.token:
                self.state_file.unlink()
        except (OSError, ValueError):
            pass

    def _monitor(self):
        try:
            while not self.done.wait(.01):
                self.controller.watch()
                age = time.monotonic() - self.started
                idle = time.monotonic() - self.last_action
                if self.controller.stopped:
                    self.log('F11 / 停止信号：输入已停止。')
                    break
                if age >= self.max_seconds or idle >= self.idle_seconds:
                    self.log('本次会话已到期，请点击“启动桥接”重新开始。')
                    break
        except Exception as exc:
            self.log('输入监护异常，已停止：' + str(exc))
        finally:
            self.stop()
            self.server.shutdown()
            self.server.server_close()
            self._cleanup_state()
            self.closed.set()

    def close(self):
        self.stop()
        for thread in self.threads:
            if thread is not threading.current_thread():
                thread.join(timeout=3)


def request(command, data=None, state_file=STATE_FILE):
    if command not in {'health', 'action', 'stop'}:
        raise ValueError('Unknown command')
    state = json.loads(Path(state_file).read_text(encoding='utf-8'))
    if type(state.get('port')) is not int or not 1 <= state['port'] <= 65535:
        raise ValueError('Invalid local bridge port')
    if not isinstance(state.get('token'), str) or len(state['token']) > 128:
        raise ValueError('Invalid local bridge token')
    body = None if command == 'health' else json.dumps(data or {}).encode()
    req = urllib.request.Request('http://127.0.0.1:' + str(state['port']) + '/' + command,
                                 data=body, headers={'Authorization': 'Bearer ' + state['token'],
                                                     'Content-Type': 'application/json'})
    # Refuse proxies and redirects so local credentials cannot be forwarded.
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    try:
        with opener.open(req, timeout=5) as response:
            return json.loads(response.read(4096))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(exc.read(4096).decode(errors='replace')) from None
    except (TimeoutError, urllib.error.URLError):
        raise RuntimeError('Outcome unknown / bridge unavailable; inspect before retrying') from None
