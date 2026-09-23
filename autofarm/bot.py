"""运行控制：F12 开始/暂停开关、F11 退出、失焦自动暂停、可打断等待。

- 启动默认暂停，F12 边沿触发切换（按一下开、再按一下关）
- 游戏窗口不在前台时一律不发键，回到前台自动继续
- 等待期间每 20ms 轮询一次热键/前台；暂停和失焦的时间不计入等待
- 移动键按住段不轮询（时长极短），避免游戏里方向键卡住
- 绑定窗口后先校验权限：游戏以管理员运行时本程序也必须提权，否则白跑
"""
import time
import queue

from .winapi import ElevationMismatch, WinApi

POLL_SECS = 0.02


class BotStopped(Exception):
    """F11 请求退出。"""


class Bot:
    def __init__(self, window_title, log=print):
        self.log = log
        self.api = WinApi()
        self.hwnd, self.window_name = self.api.find_window(window_title)
        self._check_elevation()
        self.paused = True          # 启动即暂停，等 F12
        self.quitting = False
        self.held = set()          # 已发下、还没发上的键，退出时兜底松开
        self.buff_commands = queue.Queue()
        self.buff_status = '等待开始'
        # 预热热键状态，避免启动瞬间把残留状态当成一次按下
        self._prev = {
            'f12': self.api.async_pressed('f12'),
            'f11': self.api.async_pressed('f11'),
        }
        self._fg_lost = False

    # ---------- 权限 ----------

    def _check_elevation(self):
        """游戏提权时本程序也必须提权：否则 UIPI 会静默丢掉我们发的所有按键。

        这种情况下热键、闸门、日志全都正常，游戏却完全没反应，最难排查，
        所以绑定窗口时直接拦下来报错。
        """
        target = self.api.process_elevated(self.api.window_pid(self.hwnd))
        if target and not self.api.process_elevated():
            raise ElevationMismatch(
                f'游戏窗口「{self.window_name}」以管理员身份运行，本程序没有。'
                'Windows 会静默忽略低权限程序发出的按键，挂机不会有任何反应。'
                '请关掉本程序，右键选择「以管理员身份运行」后重试。')

    # ---------- 状态 ----------

    def foreground(self):
        return self.api.get_foreground() == self.hwnd and not self.api.is_iconic(self.hwnd)

    def poll_hotkeys(self):
        for name in ('f12', 'f11'):
            down = self.api.async_pressed(name)
            if down and not self._prev[name]:
                if name == 'f12':
                    self._toggle()
                else:
                    self.quitting = True
            self._prev[name] = down

    def _toggle(self):
        self.paused = not self.paused
        if self.paused:
            self.log('[暂停] F12 已暂停')
        else:
            self.log('[运行] F12 开始')
            time.sleep(0.3)        # 恢复后留点反应时间，防手抖连按

    # ---------- 等待 ----------

    def wait(self, secs):
        """可打断等待：暂停/失焦时挂起且不计时，恢复后把剩余时间等完。"""
        remaining = float(secs)
        interrupted = False
        while remaining > 0:
            self.poll_hotkeys()
            if self.quitting:
                raise BotStopped
            if self.paused:
                interrupted = True
                time.sleep(POLL_SECS)
                continue
            if not self.foreground():
                interrupted = True
                if not self._fg_lost:
                    self.log('[保护] 游戏不在前台，挂起等待…')
                    self._fg_lost = True
                time.sleep(0.05)
                continue
            if self._fg_lost:
                self.log('[恢复] 游戏回到前台，继续')
                self._fg_lost = False
            chunk = min(POLL_SECS, remaining)
            time.sleep(chunk)
            remaining -= chunk
        return interrupted

    # ---------- 动作闸门 ----------

    def gate(self):
        """所有发键动作前的闸门：F11 退出；暂停/失焦时原地挂起，恢复运行才放行。

        这样方案循环里第一个动作（哪怕是移动）也必须等 F12 开始且游戏在前台。
        """
        interrupted = False
        while True:
            self.poll_hotkeys()
            if self.quitting:
                raise BotStopped
            if not self.paused and self.foreground():
                if self._fg_lost:
                    self.log('[恢复] 游戏回到前台，继续')
                    self._fg_lost = False
                return interrupted
            interrupted = True
            if not self.paused and not self.foreground() and not self._fg_lost:
                self.log('[保护] 游戏不在前台，挂起等待…')
                self._fg_lost = True
            time.sleep(POLL_SECS)

    # ---------- 按键 ----------

    def down(self, key):
        self.api.send_key(key, up=False)
        self.held.add(key)

    def up(self, key):
        self.api.send_key(key, up=True)
        self.held.discard(key)

    def tap(self, key, hold):
        """点按：按下后停留 hold 秒再松开（按住段不轮询，本来就只有几十毫秒）。"""
        self.gate()
        self.down(key)
        time.sleep(max(0.0, hold))
        self.up(key)

    def hold_key(self, key, secs):
        """按住（移动用）：先过闸门；整个按住段不可打断，保证一定发上抬起消息，防卡键。"""
        self.gate()
        self.down(key)
        time.sleep(max(0.0, secs))
        self.up(key)

    def release_all(self):
        for key in tuple(self.held):
            self.api.send_key(key, up=True)
        self.held.clear()

    def try_tap(self, key, hold):
        """基于刚取得的截图发键；暂停或失焦直接放弃，不等恢复后补发。"""
        self.poll_hotkeys()
        if self.quitting:
            raise BotStopped
        if self.paused or not self.foreground():
            return False
        try:
            self.down(key)
            until = time.monotonic() + hold
            while time.monotonic() < until:
                self.poll_hotkeys()
                if self.quitting:
                    raise BotStopped
                if self.paused or not self.foreground():
                    return False
                time.sleep(min(.005, max(0, until - time.monotonic())))
            return True
        finally:
            self.up(key)
