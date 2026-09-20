import ctypes as C
from ctypes import wintypes as W
import time

KEYS = {'left': 0x25, 'up': 0x26, 'right': 0x27, 'down': 0x28,
        'shift': 0x10, 'ctrl': 0x11, 'alt': 0x12, 'space': 0x20,
        'f8': 0x77, 'f9': 0x78}
KEYS.update({chr(i): i-32 for i in range(97, 123)})
KEYS.update({str(i): ord(str(i)) for i in range(10)})


class KeyboardInput(C.Structure):
    _fields_ = [('vk', W.WORD), ('scan', W.WORD), ('flags', W.DWORD),
                ('time', W.DWORD), ('extra', C.c_size_t)]


class MouseInput(C.Structure):
    _fields_ = [('dx', W.LONG), ('dy', W.LONG), ('data', W.DWORD),
                ('flags', W.DWORD), ('time', W.DWORD), ('extra', C.c_size_t)]


class InputUnion(C.Union):
    _fields_ = [('ki', KeyboardInput), ('mi', MouseInput)]


class Input(C.Structure):
    _fields_ = [('type', W.DWORD), ('value', InputUnion)]


class Desktop:
    def __init__(self, title):
        self.u = C.WinDLL('user32', use_last_error=True)
        self.u.SetProcessDPIAware()
        self.u.GetForegroundWindow.restype = W.HWND
        self.u.GetClientRect.argtypes = [W.HWND, C.POINTER(W.RECT)]
        self.u.ClientToScreen.argtypes = [W.HWND, C.POINTER(W.POINT)]
        self.u.IsWindowVisible.argtypes = [W.HWND]
        self.u.IsIconic.argtypes = [W.HWND]
        self.u.GetWindowTextW.argtypes = [W.HWND, W.LPWSTR, C.c_int]
        self.u.SendInput.argtypes = [W.UINT, C.POINTER(Input), C.c_int]
        self.u.SendInput.restype = W.UINT
        matches = []
        callback_type = C.WINFUNCTYPE(W.BOOL, W.HWND, W.LPARAM)

        @callback_type
        def callback(hwnd, _):
            text = C.create_unicode_buffer(512)
            self.u.GetWindowTextW(hwnd, text, 512)
            if self.u.IsWindowVisible(hwnd) and title.lower() in text.value.lower():
                matches.append((hwnd, text.value))
            return True

        self.u.EnumWindows(callback, 0)
        if len(matches) != 1:
            raise RuntimeError(f'窗口标题必须唯一匹配，当前结果: {matches}')
        self.hwnd = matches[0][0]
        self.held = set()

    def foreground(self):
        return self.u.GetForegroundWindow() == self.hwnd and not self.u.IsIconic(self.hwnd)

    def region(self):
        rect, point = W.RECT(), W.POINT(0, 0)
        if not self.u.GetClientRect(self.hwnd, C.byref(rect)) or not self.u.ClientToScreen(self.hwnd, C.byref(point)):
            raise RuntimeError('无法读取游戏窗口')
        if rect.right <= 0 or rect.bottom <= 0:
            raise RuntimeError('游戏窗口没有可截图区域')
        return dict(left=point.x, top=point.y, width=rect.right, height=rect.bottom)

    def pressed(self, key):
        return bool(self.u.GetAsyncKeyState(KEYS[key]) & 0x8000)

    def send(self, key, up=False):
        extended = 1 if key in ('left', 'right', 'up', 'down') else 0
        event = Input(type=1, value=InputUnion(ki=KeyboardInput(KEYS[key], 0, extended | (2 if up else 0), 0, 0)))
        if self.u.SendInput(1, C.byref(event), C.sizeof(Input)) != 1:
            raise RuntimeError('SendInput 失败；请检查窗口权限')
        if up:
            self.held.discard(key)
        else:
            self.held.add(key)

    def release(self):
        for key in tuple(self.held):
            self.send(key, up=True)

    def execute(self, action):
        if not self.foreground() or self.pressed('f9'):
            return False
        try:
            for key in action.keys:
                self.send(key)
            end = time.monotonic() + action.duration
            while time.monotonic() < end:
                if not self.foreground() or self.pressed('f9') or self.pressed('f8'):
                    return False
                time.sleep(0.01)
            return True
        finally:
            self.release()
