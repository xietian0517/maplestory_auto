"""最底层 Win32 封装：找窗口、SendInput 按键（带扫描码）、全局热键状态。

实测该怀旧服客户端（Unity IL2CPP）不处理 PostMessage 队列消息，
只认 SendInput 系统级注入，因此按键必须在游戏前台时发送（bot 的闸门已保证）。

另注意：游戏由 GameGuard 拉起、以管理员身份运行，本程序权限必须不低于它，
否则 UIPI 会静默丢弃 SendInput 的按键（返回值正常，游戏却毫无反应）。
"""
import ctypes as C
from ctypes import wintypes as W

KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002

# 进程权限查询用
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
TOKEN_QUERY = 0x0008
TOKEN_ELEVATION_CLASS = 20

# 按键名 -> Windows 虚拟键码
VK_CODES = {
    'left': 0x25, 'up': 0x26, 'right': 0x27, 'down': 0x28,
    'shift': 0xA0,          # VK_LSHIFT（扫描码 0x2A，游戏认这个）
    'ctrl': 0xA2,           # VK_LCONTROL
    'alt': 0xA4,            # VK_LMENU（左Alt=跳，扫描码 0x38）
    'space': 0x20,
    'end': 0x23, 'home': 0x24, 'pgup': 0x21, 'pgdn': 0x22,
    'insert': 0x2D, 'delete': 0x2E, 'enter': 0x0D, 'esc': 0x1B,
    'f8': 0x77, 'f9': 0x78, 'f11': 0x7A, 'f12': 0x7B,
}
VK_CODES.update({chr(c): c - 32 for c in range(97, 123)})        # a-z
VK_CODES.update({str(d): ord(str(d)) for d in range(10)})        # 0-9

# 扩展键：方向键、End/Home/Insert/Delete/PgUp/PgDn，SendInput 的 flags 要置扩展位
EXTENDED_KEYS = {'left', 'up', 'right', 'down', 'end', 'home',
                 'insert', 'delete', 'pgup', 'pgdn'}


class _KeyboardInput(C.Structure):
    _fields_ = [('vk', W.WORD), ('scan', W.WORD), ('flags', W.DWORD),
                ('time', W.DWORD), ('extra', C.c_size_t)]


class _MouseInput(C.Structure):
    _fields_ = [('dx', W.LONG), ('dy', W.LONG), ('data', W.DWORD),
                ('flags', W.DWORD), ('time', W.DWORD), ('extra', C.c_size_t)]


class _InputUnion(C.Union):
    _fields_ = [('ki', _KeyboardInput), ('mi', _MouseInput)]


class _Input(C.Structure):
    _fields_ = [('type', W.DWORD), ('value', _InputUnion)]


class WindowBindError(RuntimeError):
    pass


class ElevationMismatch(RuntimeError):
    """游戏进程权限高于本程序：UIPI 会丢弃本程序注入的按键。"""


class _TokenElevation(C.Structure):
    _fields_ = [('elevated', W.DWORD)]


class WinApi:
    def __init__(self):
        u = self.u = C.WinDLL('user32', use_last_error=True)
        self.k32 = C.WinDLL('kernel32', use_last_error=True)
        self.adv = C.WinDLL('advapi32', use_last_error=True)
        self._enum_proc = C.WINFUNCTYPE(W.BOOL, W.HWND, W.LPARAM)
        u.EnumWindows.argtypes = [self._enum_proc, W.LPARAM]
        u.EnumWindows.restype = W.BOOL
        u.GetWindowTextW.argtypes = [W.HWND, W.LPWSTR, C.c_int]
        u.IsWindowVisible.argtypes = [W.HWND]
        u.IsIconic.argtypes = [W.HWND]
        u.GetForegroundWindow.restype = W.HWND
        u.MapVirtualKeyW.argtypes = [W.UINT, W.UINT]
        u.MapVirtualKeyW.restype = W.UINT
        u.SendInput.argtypes = [W.UINT, C.POINTER(_Input), C.c_int]
        u.SendInput.restype = W.UINT
        u.GetAsyncKeyState.argtypes = [C.c_int]
        u.GetAsyncKeyState.restype = C.c_short
        u.GetWindowThreadProcessId.argtypes = [W.HWND, C.POINTER(W.DWORD)]
        u.GetClientRect.argtypes = [W.HWND, C.POINTER(W.RECT)]
        u.ClientToScreen.argtypes = [W.HWND, C.POINTER(W.POINT)]
        self.k32.OpenProcess.argtypes = [W.DWORD, W.BOOL, W.DWORD]
        self.k32.OpenProcess.restype = W.HANDLE
        self.k32.CloseHandle.argtypes = [W.HANDLE]
        self.k32.GetCurrentProcess.restype = W.HANDLE
        self.adv.OpenProcessToken.argtypes = [W.HANDLE, W.DWORD, C.POINTER(W.HANDLE)]
        self.adv.GetTokenInformation.argtypes = [W.HANDLE, C.c_int, C.c_void_p,
                                                W.DWORD, C.POINTER(W.DWORD)]

    def window_pid(self, hwnd):
        """窗口属于哪个进程。"""
        pid = W.DWORD()
        self.u.GetWindowThreadProcessId(hwnd, C.byref(pid))
        return pid.value

    def client_rect(self, hwnd):
        """客户区在屏幕上的位置与大小（截图用，不含标题栏和边框）。"""
        rect, origin = W.RECT(), W.POINT(0, 0)
        if not self.u.GetClientRect(hwnd, C.byref(rect)) or \
                not self.u.ClientToScreen(hwnd, C.byref(origin)):
            raise RuntimeError('读不到游戏窗口客户区')
        if rect.right <= 0 or rect.bottom <= 0:
            raise RuntimeError('游戏窗口没有可截图区域（是否最小化了）')
        return dict(left=origin.x, top=origin.y,
                    width=rect.right, height=rect.bottom)

    def process_elevated(self, pid=None):
        """进程是否以管理员运行。pid 为空表示本进程；查询失败返回 None（未知）。"""
        own = pid is None
        handle = (self.k32.GetCurrentProcess() if own
                  else self.k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid))
        if not handle:
            return None
        try:
            token = W.HANDLE()
            if not self.adv.OpenProcessToken(handle, TOKEN_QUERY, C.byref(token)):
                return None
            try:
                info = _TokenElevation()
                size = W.DWORD()
                if not self.adv.GetTokenInformation(token, TOKEN_ELEVATION_CLASS,
                                                    C.byref(info), C.sizeof(info),
                                                    C.byref(size)):
                    return None
                return bool(info.elevated)
            finally:
                self.k32.CloseHandle(token)
        finally:
            if not own:
                self.k32.CloseHandle(handle)

    def find_window(self, title_part):
        """按标题关键字找唯一可见窗口，返回 (hwnd, 完整标题)。"""
        matches = []

        def _cb(hwnd, _):
            buf = C.create_unicode_buffer(512)
            self.u.GetWindowTextW(hwnd, buf, 512)
            if self.u.IsWindowVisible(hwnd) and title_part.lower() in buf.value.lower():
                matches.append((hwnd, buf.value))
            return True

        self.u.EnumWindows(self._enum_proc(_cb), 0)
        if not matches:
            raise WindowBindError(f'找不到标题包含 "{title_part}" 的可见窗口，请先开游戏并改 farm.py 里的 WINDOW_TITLE')
        if len(matches) > 1:
            names = '、'.join(t for _, t in matches)
            raise WindowBindError(f'匹配到多个窗口（{names}），请把 WINDOW_TITLE 改成更唯一的关键字')
        return matches[0]

    def get_foreground(self):
        return self.u.GetForegroundWindow()

    def is_iconic(self, hwnd):
        return bool(self.u.IsIconic(hwnd))

    def async_pressed(self, name):
        """全局热键是否按下（F11/F12 用），不要求游戏在前台。"""
        return bool(self.u.GetAsyncKeyState(VK_CODES[name]) & 0x8000)

    def send_key(self, name, up):
        """SendInput 注入一次按键（发往当前前台窗口，调用方必须保证游戏在前台）。"""
        vk = VK_CODES[name]
        scan = self.u.MapVirtualKeyW(vk, 0)
        flags = (KEYEVENTF_EXTENDEDKEY if name in EXTENDED_KEYS else 0)
        if up:
            flags |= KEYEVENTF_KEYUP
        event = _Input(type=1, value=_InputUnion(ki=_KeyboardInput(vk, scan, flags, 0, 0)))
        if self.u.SendInput(1, C.byref(event), C.sizeof(_Input)) != 1:
            raise RuntimeError(f'SendInput 失败：{C.get_last_error()}')
