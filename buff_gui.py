# -*- coding: utf-8 -*-
"""独立加Buff工具（GUI 版）：图形界面管理多个 Buff 槽位，定时按键，不共享CD。

运行：  .\\.venv\\Scripts\\python.exe buff_gui.py
打包：  .\\.venv\\Scripts\\python.exe -m PyInstaller --onefile --windowed --uac-admin --name MapleBuffGUI buff_gui.py

- 每个槽位：按键名 + 间隔(秒,随机区间) + 按后停顿；「+添加」可加任意多个，「×」删除
- 槽位之间各自独立计时，互不影响（不共享CD）
- 启动后是暂停态：切到游戏按 F12 开始/暂停，F11 退出，界面按钮等效
- 游戏不在前台自动挂起；游戏是管理员权限时本程序也必须管理员运行（exe 已内置提权）
- 配置保存在 exe/脚本同目录 buff_gui_config.json，下次打开自动加载
"""
import ctypes
import json
import os
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox

from autofarm.bot import Bot, BotStopped
from autofarm.blocks import Interval, rnd
from autofarm.winapi import ElevationMismatch, WindowBindError

CONFIG_FILE = 'buff_gui_config.json'
KEY_HOLD_SECS = (0.030, 0.080)     # Buff键按下停留时长
POLL_SECS = 0.1                    # 空闲轮询间隔
MAX_LOG_LINES = 300

DEFAULT_SLOTS = [                  # 默认两个槽位：home / ins
    {'key': 'home', 'lo': '60', 'hi': '90', 'pmin': '0.3', 'pmax': '0.6'},
    {'key': 'ins', 'lo': '120', 'hi': '180', 'pmin': '0.3', 'pmax': '0.6'},
]


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('冒险岛加Buff助手')
        self.resizable(False, False)
        self.bot = None
        self.alive = False
        self._closing = False
        self.q = queue.Queue()
        self.slot_rows = []        # [{'frame':..,'key':..,'lo':..,'hi':..,'pmin':..,'pmax':..}]
        self._build()
        self._load()
        self.after(120, self._drain)
        self.protocol('WM_DELETE_WINDOW', self._on_close)

    # ---------- 界面 ----------

    def _build(self):
        top = ttk.LabelFrame(self, text='窗口')
        top.grid(row=0, column=0, sticky='we', padx=8, pady=(8, 2))
        ttk.Label(top, text='窗口标题').grid(row=0, column=0, sticky='e', padx=4, pady=2)
        self.win_var = tk.StringVar(value='冒险岛怀旧服')
        ttk.Entry(top, width=28, textvariable=self.win_var).grid(row=0, column=1, sticky='w', padx=4)

        slots = ttk.LabelFrame(self, text='Buff槽位（键名 | 间隔秒 | 按后停顿秒）——不共享CD')
        slots.grid(row=1, column=0, sticky='we', padx=8, pady=2)
        header = ttk.Label(slots, text='键名        间隔min   间隔max   停顿min   停顿max')
        header.grid(row=0, column=0, sticky='w', padx=4)
        self.slots_frame = ttk.Frame(slots)
        self.slots_frame.grid(row=1, column=0, sticky='we', padx=4)
        ttk.Button(slots, text='+添加一个Buff键', command=lambda: self._add_slot_row()).grid(
            row=2, column=0, sticky='w', padx=4, pady=4)

        ctrl = ttk.Frame(self)
        ctrl.grid(row=2, column=0, sticky='we', padx=8, pady=4)
        self.btn_start = ttk.Button(ctrl, text='启动（挂后台等F12）', command=self.on_start)
        self.btn_start.pack(side='left', padx=4)
        self.btn_pause = ttk.Button(ctrl, text='开始', command=self.on_pause, state='disabled')
        self.btn_pause.pack(side='left', padx=4)
        self.btn_stop = ttk.Button(ctrl, text='停止', command=self.on_stop, state='disabled')
        self.btn_stop.pack(side='left', padx=4)
        self.status = tk.StringVar(value='未运行')
        ttk.Label(ctrl, textvariable=self.status, foreground='#0a7').pack(side='left', padx=10)

        logf = ttk.LabelFrame(self, text='日志')
        logf.grid(row=3, column=0, sticky='we', padx=8, pady=(2, 8))
        self.log_text = tk.Text(logf, height=10, width=60, state='disabled', font=('Consolas', 9))
        self.log_text.pack(side='left', fill='both', expand=True)
        sb = ttk.Scrollbar(logf, command=self.log_text.yview)
        sb.pack(side='right', fill='y')
        self.log_text['yscrollcommand'] = sb.set
        self._lines = 0
        for slot in DEFAULT_SLOTS:
            self._add_slot_row(**slot)

    def _add_slot_row(self, key='home', lo='60', hi='90', pmin='0.3', pmax='0.6'):
        f = ttk.Frame(self.slots_frame)
        widgets = {}
        for i, (name, width, value) in enumerate(
                [('key', 7, key), ('lo', 6, lo), ('hi', 6, hi), ('pmin', 6, pmin), ('pmax', 6, pmax)]):
            widgets[name] = tk.StringVar(value=str(value))
            ttk.Entry(f, width=width, textvariable=widgets[name]).grid(row=0, column=i, padx=3)
        btn = ttk.Button(f, text='×', width=2,
                         command=lambda: self._remove_slot(f))
        btn.grid(row=0, column=5, padx=3)
        f.pack(anchor='w', pady=1)
        self.slot_rows.append({'frame': f, **widgets})

    def _remove_slot(self, frame):
        self.slot_rows = [r for r in self.slot_rows if r['frame'] is not frame]
        frame.destroy()

    # ---------- 配置持久化 ----------

    def _slots(self):
        rows = []
        for r in self.slot_rows:
            rows.append({k: v.get().strip() for k, v in r.items() if k != 'frame'})
        return [r for r in rows if r['key']]

    def _save(self):
        try:
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump({'window_title': self.win_var.get(),
                           'slots': self._slots()}, f, ensure_ascii=False, indent=1)
        except OSError:
            pass

    def _load(self):
        if not os.path.exists(CONFIG_FILE):
            return
        try:
            with open(CONFIG_FILE, encoding='utf-8') as f:
                data = json.load(f)
        except (OSError, ValueError):
            return
        self.win_var.set(data.get('window_title', self.win_var.get()))
        slots = data.get('slots') or []
        if slots:
            for r in list(self.slot_rows):
                r['frame'].destroy()
            self.slot_rows = []
            for s in slots:
                self._add_slot_row(**{k: s.get(k, v) for k, v in DEFAULT_SLOTS[0].items()})

    # ---------- 动作 ----------

    def _log(self, msg):
        stamp = time.strftime('%H:%M:%S')
        self.log_text['state'] = 'normal'
        self.log_text.insert('end', f'[{stamp}] {msg}\n')
        self._lines += 1
        if self._lines > MAX_LOG_LINES:
            self.log_text.delete('1.0', '2.0')
            self._lines -= 1
        self.log_text.see('end')
        self.log_text['state'] = 'disabled'

    def on_start(self):
        if self.alive:
            return
        slots = self._slots()
        if not slots:
            messagebox.showwarning('没有槽位', '先添加至少一个Buff键')
            return
        try:
            for s in slots:
                lo, hi = float(s['lo']), float(s['hi'])
                if lo > hi:
                    raise ValueError(f'键 {s["key"]} 的间隔最小值大于最大值')
        except ValueError as e:
            messagebox.showerror('参数错误', f'参数有误：{e}')
            return
        self._save()
        self.alive = True
        self.btn_start['state'] = 'disabled'
        self.btn_pause['state'] = 'normal'
        self.btn_stop['state'] = 'normal'
        self.btn_pause['text'] = '开始'
        self.status.set('启动中…')
        threading.Thread(target=self._worker, args=(slots,), daemon=True).start()

    def on_pause(self):
        if self.bot is None:
            return
        self.bot.paused = not self.bot.paused
        if self.bot.paused:
            self.q.put(('log', '[暂停] 手动暂停'))
            self.btn_pause['text'] = '开始'
        else:
            self.q.put(('log', '[运行] 手动开始'))
            self.btn_pause['text'] = '暂停'

    def on_stop(self):
        if self.bot is not None:
            self.bot.quitting = True
            self.status.set('正在停止…')

    def _worker(self, slots):
        q = self.q
        bot = None
        try:
            bot = Bot(self.win_var.get().strip() or '冒险岛怀旧服',
                      log=lambda m: q.put(('log', m)))
            self.bot = bot
            api = bot.api
            q.put(('log', f'绑定窗口：{bot.window_name}（句柄 {bot.hwnd}）'))
            q.put(('log', f'权限：本程序{"已提权" if api.process_elevated() else "未提权"}，'
                          f'游戏{"已提权" if api.process_elevated(api.window_pid(bot.hwnd)) else "未提权"}；'
                          f'游戏当前{"在前台" if bot.foreground() else "不在前台"}'))
            for s in slots:
                q.put(('log', f'Buff {s["key"]}：每 {s["lo"]}~{s["hi"]} 秒一次'))
            q.put(('log', '已暂停：切到游戏按 F12 或点「开始」；F11/「停止」退出。'))
            timers = [(s['key'], Interval((float(s['lo']), float(s['hi']))),
                       (float(s['pmin'] or 0.3), float(s['pmax'] or 0.6))) for s in slots]
            last_status = time.monotonic()
            while True:
                for key, timer, pause in timers:
                    if timer.due():
                        hold = rnd(KEY_HOLD_SECS)
                        bot.log(f'[Buff] 按 {key}（按住 {hold * 1000:.0f}ms，'
                                f'之后停 {rnd(pause):.2f}s）——前台={bot.foreground()}')
                        bot.tap(key, hold)
                        bot.wait(rnd(pause))
                now = time.monotonic()
                if now - last_status >= 30:      # 每 30 秒报一次各槽位倒计时
                    last_status = now
                    left = '  '.join(f'{key}还有{max(0, t.next_at - now):.0f}s'
                                     for key, t, _ in timers)
                    state = '运行中' if not bot.paused else '已暂停(F12继续)'
                    if not bot.foreground():
                        state += '，游戏不在前台'
                    bot.log(f'[状态] {state} | {left}')
                bot.wait(POLL_SECS)
        except BotStopped:
            q.put(('log', '[退出] 收到停止请求。'))
        except ElevationMismatch as e:
            q.put(('log', f'[错误] {e}'))
            q.put(('elevation', None))
        except WindowBindError as e:
            q.put(('log', f'[错误] {e}'))
        except Exception as e:
            q.put(('log', f'[错误] {type(e).__name__}: {e}'))
        finally:
            if bot is not None:
                bot.release_all()
                q.put(('log', '已兜底松开所有按键。'))
            q.put(('done', None))

    def _drain(self):
        if self._closing:
            return
        try:
            while not self._closing:
                kind, payload = self.q.get_nowait()
                if kind == 'log':
                    self._log(payload)
                elif kind == 'elevation':
                    self._offer_elevation()
                elif kind == 'done':
                    self._on_done()
        except queue.Empty:
            pass
        if self._closing:
            return
        if self.alive and self.bot is not None:
            self.status.set('已暂停（F12/按钮继续）' if self.bot.paused else '定时Buff运行中…')
        self.after(120, self._drain)

    def _on_done(self):
        self.alive = False
        self.bot = None
        self.status.set('已停止')
        self.btn_start['state'] = 'normal'
        self.btn_pause['state'] = 'disabled'
        self.btn_pause['text'] = '开始'
        self.btn_stop['state'] = 'disabled'

    def _on_close(self):
        self._closing = True
        if self.bot is not None:
            self.bot.quitting = True
            try:
                self.bot.release_all()
            except Exception:
                pass
        self.destroy()

    # ---------- 提权 ----------

    def _offer_elevation(self):
        msg = ('游戏以管理员身份运行，本程序没有，发出去的按键会被 Windows 静默丢弃。\n'
               '需要以管理员身份重新启动本程序。\n\n现在重启吗？')
        if messagebox.askyesno('需要管理员权限', msg):
            self._relaunch_as_admin()

    def _relaunch_as_admin(self):
        args = ' '.join(f'"{a}"' for a in sys.argv[1:]) or None
        rc = ctypes.windll.shell32.ShellExecuteW(None, 'runas', sys.executable, args, None, 1)
        if rc <= 32:
            messagebox.showwarning('提权未完成',
                                   '以管理员身份重启被取消或失败，'
                                   '请手动右键程序选择「以管理员身份运行」。')
            return
        self._on_close()


def main():
    if getattr(sys, 'frozen', False):
        os.chdir(os.path.dirname(sys.executable))
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass
    App().mainloop()


if __name__ == '__main__':
    main()
