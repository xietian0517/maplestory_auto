"""持续按键与独立松键监护：截图耗时或阻塞也不能无限持键。"""
import threading
import time

from .bot import BotStopped


class HeldInput:
    def __init__(self, bot):
        self.bot = bot
        self.key = None
        self.deadline = 0.0
        self.epoch = 0
        self.blocked = False
        self.lock = threading.RLock()
        self.stopping = threading.Event()
        self.thread = None

    def _clear(self):
        if self.key is not None:
            key, self.key = self.key, None
            self.bot.up(key)

    def clear(self):
        with self.lock:
            self._clear()

    def _allowed(self):
        return not self.bot.paused and not self.bot.quitting and self.bot.foreground()

    def check(self):
        with self.lock:
            allowed = self._allowed()
            if not allowed and not self.blocked:
                self.epoch += 1
            self.blocked = not allowed
            if not allowed or time.monotonic() >= self.deadline:
                self._clear()

    def apply(self, key, deadline, epoch):
        """同方向续按只续期，不重复发送 keydown/keyup。"""
        self.bot.poll_hotkeys()
        if self.bot.quitting:
            raise BotStopped
        with self.lock:
            if epoch != self.epoch or not self._allowed() or time.monotonic() >= deadline:
                self._clear()
                return False
            if self.key != key:
                self._clear()
                self.bot.down(key)
                self.key = key
            self.deadline = deadline
            return True

    def _watch(self):
        try:
            while not self.stopping.wait(.005):
                self.check()
        except Exception:
            self.bot.quitting = True
            self.clear()

    def __enter__(self):
        self.thread = threading.Thread(target=self._watch, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stopping.set()
        self.thread.join()
        self.clear()
