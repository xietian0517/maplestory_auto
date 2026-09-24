"""Game-only elevated bridge GUI and standard-permission command client."""
import argparse
import ctypes as C
from ctypes import wintypes as W
import json
import os
from pathlib import Path
import queue
import threading
import time

from autofarm.bridge_service import BridgeService, STATE_FILE, make_action, request
from autofarm.winapi import WinApi

VERSION = '0.1.0'
GAME = Path(r'F:\mxd\冒险岛online\mxdclassic\Maplestory_Classic.exe')
GAME_TITLE = '冒险岛怀旧服'


class GameAdapter:
    def __init__(self):
        self.api = WinApi()
        if not self.api.process_elevated():
            raise RuntimeError('请双击发行版 EXE，并在 Windows 提示中允许管理员运行。')
        self.api.k32.QueryFullProcessImageNameW.argtypes = [
            W.HANDLE, W.DWORD, W.LPWSTR, C.POINTER(W.DWORD)]
        self.api.k32.QueryFullProcessImageNameW.restype = W.BOOL
        self.api.u.GetWindowTextW.restype = C.c_int
        self.bound = {}

    def validate_target(self, hwnd):
        title = C.create_unicode_buffer(512)
        self.api.u.GetWindowTextW(hwnd, title, len(title))
        if title.value != GAME_TITLE:
            raise ValueError('目标窗口不是冒险岛怀旧服。')
        pid = self.api.window_pid(hwnd)
        handle = self.api.k32.OpenProcess(0x1000, False, pid)
        if not handle:
            raise ValueError('无法核实游戏进程。')
        try:
            path = C.create_unicode_buffer(32768)
            length = W.DWORD(len(path))
            if not self.api.k32.QueryFullProcessImageNameW(handle, 0, path, C.byref(length)):
                raise ValueError('无法核实游戏程序路径。')
            if os.path.normcase(path.value) != os.path.normcase(str(GAME)):
                raise ValueError('该窗口不属于指定的冒险岛客户端。')
            self.bound[hwnd] = pid
        finally:
            self.api.k32.CloseHandle(handle)

    def find_game(self):
        hwnd, title = self.api.find_window(GAME_TITLE)
        self.validate_target(hwnd)
        return hwnd

    def is_target_foreground(self, hwnd):
        return (self.api.get_foreground() == hwnd and not self.api.is_iconic(hwnd)
                and self.bound.get(hwnd) == self.api.window_pid(hwnd))

    def emergency_pressed(self):
        return self.api.async_pressed('f11')

    def send_key(self, key, up):
        self.api.send_key(key, up)


class SingleInstance:
    """One bridge per interactive Windows session; duplicate launch is harmless."""
    def __init__(self):
        self.k32 = C.WinDLL('kernel32', use_last_error=True)
        self.k32.CreateMutexW.argtypes = [C.c_void_p, W.BOOL, W.LPCWSTR]
        self.k32.CreateMutexW.restype = W.HANDLE
        self.k32.CloseHandle.argtypes = [W.HANDLE]
        self.handle = self.k32.CreateMutexW(None, False, r'Local\MapleInputBridge-v1')
        if not self.handle:
            raise RuntimeError('无法建立桥接单实例锁。')
        if C.get_last_error() == 183:
            self.close()
            raise RuntimeError('输入桥接已经打开，请使用现有窗口。')

    def close(self):
        if self.handle:
            self.k32.CloseHandle(self.handle)
            self.handle = None


class BridgeWindow:
    def __init__(self, root, adapter, autostart=True, service_factory=BridgeService):
        import tkinter as tk
        from tkinter import ttk
        self.root = root
        self.adapter = adapter
        self.service_factory = service_factory
        self.service = None
        self.messages = queue.Queue()
        self.pending = None
        self.pending_at = 0
        self.worker = None
        self.root.title('冒险岛输入桥接 v' + VERSION)
        self.root.geometry('660x560')
        self.root.minsize(640, 530)
        self.root.protocol('WM_DELETE_WINDOW', self.close)
        style = ttk.Style()
        style.configure('Title.TLabel', font=('Microsoft YaHei UI', 17, 'bold'))
        style.configure('TLabel', font=('Microsoft YaHei UI', 10))
        frame = ttk.Frame(root, padding=20)
        frame.pack(fill='both', expand=True)
        ttk.Label(frame, text='冒险岛 · 输入桥接', style='Title.TLabel').pack(anchor='w')
        ttk.Label(frame, text='由你或 Codex 逐次下达按键；不会自行打怪。').pack(anchor='w', pady=(5, 15))
        self.status = tk.StringVar(value='尚未启动')
        self.game_status = tk.StringVar(value='等待游戏')
        ttk.Label(frame, textvariable=self.status).pack(anchor='w')
        ttk.Label(frame, textvariable=self.game_status).pack(anchor='w', pady=(5, 12))
        controls = ttk.Frame(frame)
        controls.pack(fill='x')
        self.start_button = ttk.Button(controls, text='启动桥接', command=self.start)
        self.start_button.pack(side='left')
        ttk.Button(controls, text='立即停止（F11）', command=self.stop).pack(side='left', padx=10)
        box = ttk.LabelFrame(frame, text='手动验证', padding=12)
        box.pack(fill='x', pady=15)
        ttk.Label(box, text='点击后有 3 秒切回游戏，随后只执行一次。').pack(anchor='w')
        self.countdown = tk.StringVar(value='请先进入角色，并关闭原挂机助手。')
        ttk.Label(box, textvariable=self.countdown).pack(anchor='w', pady=(5, 10))
        buttons = ttk.Frame(box)
        buttons.pack(fill='x')
        self.test_buttons = []
        for label, keys, ms in [('右移 0.2 秒', ['right'], 200),
                                ('跳跃 Alt', ['alt'], 100),
                                ('攻击 Shift', ['shift'], 100)]:
            button = ttk.Button(buttons, text=label,
                                command=lambda k=keys, d=ms: self.schedule_test(k, d))
            button.pack(side='left', padx=(0, 8))
            self.test_buttons.append(button)
        ttk.Label(frame, text='失焦会松键；单次最长 2 秒；空闲 5 分钟或运行 15 分钟后停止。',
                  wraplength=605).pack(anchor='w', pady=(0, 10))
        self.log_widget = tk.Text(frame, height=6, font=('Microsoft YaHei UI', 9),
                                  wrap='word', state='disabled')
        self.log_widget.pack(fill='both', expand=True)
        if autostart:
            self.start()
        self.tick_id = self.root.after(100, self.tick)

    def log(self, text):
        self.messages.put(text)

    def start(self):
        if self.service and not self.service.closed.is_set():
            return
        try:
            self.service = self.service_factory(self.adapter, log=self.log).start()
            self.log('桥接已启动。仅允许指定游戏；等待操作。')
        except Exception as exc:
            self.log('启动失败：' + str(exc))
            self.service = None

    def stop(self):
        if self.pending is not None:
            self.root.after_cancel(self.pending)
            self.pending = None
        self.countdown.set('已停止；需要继续时点击“启动桥接”。')
        if self.service:
            self.service.stop()
        self.log('已请求停止，所有按住的键会被释放。')

    def schedule_test(self, keys, ms):
        if not self.service or self.service.controller.stopped or self.pending is not None:
            return
        if self.worker and self.worker.is_alive():
            return
        try:
            hwnd = self.adapter.find_game()
        except Exception as exc:
            self.log('暂不能测试：' + str(exc))
            return
        service = self.service
        self.pending_at = time.monotonic() + 3
        self.log('准备单次测试：' + ' + '.join(keys) + '。请切回游戏。')

        def send():
            self.pending = None
            if service.controller.stopped:
                return
            self.countdown.set('测试执行中；请观察游戏。')
            def work():
                try:
                    service.execute(make_action(hwnd, keys, ms))
                except Exception as exc:
                    self.log('未完成测试：' + str(exc))
            self.worker = threading.Thread(target=work, daemon=True)
            self.worker.start()
        self.pending = self.root.after(3000, send)

    def tick(self):
        while True:
            try:
                message = self.messages.get_nowait()
            except queue.Empty:
                break
            self.log_widget.configure(state='normal')
            self.log_widget.insert('end', time.strftime('%H:%M:%S ') + message + '\n')
            if int(self.log_widget.index('end-1c').split('.')[0]) > 100:
                self.log_widget.delete('1.0', '20.0')
            self.log_widget.see('end')
            self.log_widget.configure(state='disabled')
        live = self.service is not None and not self.service.controller.stopped
        finished = self.service is None or self.service.closed.is_set()
        self.start_button.configure(state='normal' if finished else 'disabled')
        busy = self.pending is not None or (self.worker and self.worker.is_alive())
        for button in self.test_buttons:
            button.configure(state='normal' if live and not busy else 'disabled')
        if live:
            remaining = self.service.health()['remaining_seconds']
            self.status.set('桥接运行中 · 管理员输入已就绪 · 会话剩余 {}分{}秒'.format(*divmod(remaining, 60)))
        else:
            self.status.set('桥接已停止' if finished else '正在松键并停止…')
        if self.pending is not None:
            remaining = max(1, int(self.pending_at - time.monotonic()) + 1)
            self.countdown.set('{} 秒后测试，请切回游戏窗口。'.format(remaining))
        elif self.worker and not self.worker.is_alive():
            self.countdown.set('单次测试已结束；请以游戏实际反应为准。')
            self.worker = None
        try:
            hwnd = self.adapter.find_game()
            foreground = self.adapter.is_target_foreground(hwnd)
            self.game_status.set('游戏已找到 · ' + ('当前在前台' if foreground else '请切回游戏后再发送按键'))
        except Exception:
            self.game_status.set('等待游戏：请打开冒险岛并进入角色。')
        self.tick_id = self.root.after(250, self.tick)

    def close(self):
        self.root.after_cancel(self.tick_id)
        self.stop()
        if self.service:
            self.service.close()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', nargs='?', default='gui',
                        choices=['gui', 'serve', 'health', 'action', 'stop'])
    parser.add_argument('--hwnd', type=int)
    parser.add_argument('--keys', nargs='+')
    parser.add_argument('--ms', type=int, default=100)
    args = parser.parse_args()
    if args.command in {'gui', 'serve'}:
        instance = None
        try:
            instance = SingleInstance()
            adapter = GameAdapter()
            if args.command == 'gui':
                import tkinter as tk
                root = tk.Tk()
                BridgeWindow(root, adapter)
                root.mainloop()
            else:
                service = BridgeService(adapter).start()
                try:
                    service.closed.wait()
                finally:
                    service.close()
        except Exception as exc:
            if args.command == 'gui':
                C.windll.user32.MessageBoxW(None, str(exc), '冒险岛输入桥接', 0x10)
            else:
                raise
        finally:
            if instance:
                instance.close()
    else:
        if args.command == 'action' and (args.hwnd is None or not args.keys):
            parser.error('action requires --hwnd and --keys')
        try:
            data = make_action(args.hwnd, args.keys, args.ms) if args.command == 'action' else None
            print(json.dumps(request(args.command, data), ensure_ascii=False))
        except (OSError, ValueError, RuntimeError) as exc:
            parser.exit(1, str(exc) + '\n')


if __name__ == '__main__':
    main()

