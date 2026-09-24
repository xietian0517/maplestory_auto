"""Bounded game-only input; independent of the farming plans."""
import math
import threading
import time

ALLOWED_KEYS = frozenset({
    'left', 'right', 'up', 'down', 'shift', 'alt', 'ctrl', 'esc',
    'insert', 'delete', 'home', 'end', 'pgup', 'pgdn',
})


def validate_action(data, now=None):
    now = time.time() if now is None else now
    if not isinstance(data, dict) or set(data) != {'id', 'issued_at', 'hwnd', 'keys', 'ms'}:
        raise ValueError('Unexpected action fields')
    ident = data['id']
    if not isinstance(ident, str) or not 8 <= len(ident) <= 64:
        raise ValueError('Invalid action id')
    stamp = data['issued_at']
    if type(stamp) not in (int, float) or not math.isfinite(stamp) or not -1 <= now - stamp <= 5:
        raise ValueError('Action expired')
    if type(data['hwnd']) is not int or not 0 < data['hwnd'] < 2**64:
        raise ValueError('Invalid window')
    keys = data['keys']
    if not isinstance(keys, list) or not 1 <= len(keys) <= 3:
        raise ValueError('Use one to three keys')
    if any(not isinstance(key, str) or key not in ALLOWED_KEYS for key in keys):
        raise ValueError('Key not allowed')
    if len(set(keys)) != len(keys) or {'left', 'right'} <= set(keys) or {'up', 'down'} <= set(keys):
        raise ValueError('Conflicting or duplicate keys')
    if len(keys) > 1 and not set(keys) <= {'left', 'right', 'up', 'down', 'shift', 'alt'}:
        raise ValueError('Only movement, jump and attack may be combined')
    if type(data['ms']) is not int or not 30 <= data['ms'] <= 2000:
        raise ValueError('Hold must be 30..2000 ms')
    return data


class InputController:
    def __init__(self, adapter):
        self.adapter = adapter
        self.lock = threading.RLock()
        self.action_lock = threading.Lock()
        self.held = set()
        self.target = None
        self.stopped = False
        self.interrupted = False
        self.deadline = 0
        self.seen = set()

    def _release(self):
        # Attempt every release even if one send fails.
        errors = []
        for key in tuple(self.held):
            try:
                self.adapter.send_key(key, up=True)
                self.held.remove(key)
            except Exception as exc:
                errors.append(str(exc))
        return errors

    def stop(self):
        with self.lock:
            self.stopped = True
            return self._release()

    def watch(self):
        with self.lock:
            if self.adapter.emergency_pressed():
                self.stopped = True
            interrupted = self.stopped or (self.held and not self.adapter.is_target_foreground(self.target))
            if self.held and (interrupted or time.monotonic() >= self.deadline):
                self.interrupted = self.interrupted or bool(interrupted)
                errors = self._release()
                if errors:
                    self.stopped = True

    def execute(self, data):
        validate_action(data)
        if not self.action_lock.acquire(blocking=False):
            raise ValueError('An action is already running')
        start = time.monotonic()
        try:
            with self.lock:
                if self.stopped:
                    raise ValueError('Bridge stopped')
                if data['id'] in self.seen:
                    raise ValueError('Duplicate action; never replay automatically')
                if len(self.seen) >= 1000:
                    raise ValueError('Session action limit reached')
                self.seen.add(data['id'])
                self.adapter.validate_target(data['hwnd'])
                validate_action(data)  # Target inspection must not make an old request runnable.
                if not self.adapter.is_target_foreground(data['hwnd']):
                    raise ValueError('Game must be foreground; no keys sent')
                if self.adapter.emergency_pressed():
                    self.stopped = True
                    raise ValueError('F11 pressed; stopped')
                self.target = data['hwnd']
                self.interrupted = False
                self.deadline = time.monotonic() + data['ms'] / 1000
                for key in data['keys']:
                    if not self.adapter.is_target_foreground(self.target):
                        self.interrupted = True
                        break
                    # Track before SendInput so a partial failure also releases it.
                    self.held.add(key)
                    self.adapter.send_key(key, up=False)
            while time.monotonic() < self.deadline:
                self.watch()
                with self.lock:
                    if self.interrupted or self.stopped:
                        break
                time.sleep(.005)
            return {'ok': True, 'interrupted': self.interrupted or self.stopped,
                    'elapsed_ms': round((time.monotonic() - start) * 1000)}
        finally:
            with self.lock:
                errors = self._release()
                if errors:
                    self.stopped = True
                self.target = None
            self.action_lock.release()
            if errors:
                raise RuntimeError('Key release failed: ' + '; '.join(errors))
