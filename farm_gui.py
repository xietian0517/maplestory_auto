# -*- coding: utf-8 -*-
"""冒险岛挂机 GUI 版：图形界面调参数、按钮控制、日志显示、配置自动保存。

运行：  .\\.venv\\Scripts\\python.exe farm_gui.py
打包：  .\\.venv\\Scripts\\pyinstaller --onefile --windowed --name MapleFarmGUI farm_gui.py

F12 开始/暂停、F11 退出仍然有效；界面按钮与热键等效。
配置保存在 exe/脚本同目录 gui_config.json，下次启动自动加载。

游戏由 GameGuard 拉起，是管理员权限；本程序也必须以管理员运行，
否则 Windows(UIPI) 会静默丢弃按键，界面看着正常但游戏毫无反应。
打包时 spec 里已加 uac_admin=True，exe 会自动请求提权。

「截屏标定」用一次就够：切回游戏，3 秒后自动截图，先用鼠标框住自己的名字牌
（如 CatApril），再框住平台可站范围（木板左右两端）。之后 vision_jump 方案
每隔几秒截一张图，用这个名字牌找到主角在左半边还是右半边，就一直朝反方向打，
跨过中线才换方向；离两端太近时先多挪一段回中间，不会走出平台掉下去。
点「试一下」可以先验证识别。
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

import cv2

from farm import Config
from autofarm.bot import Bot, BotStopped
from autofarm import plans
from autofarm import vision as V
from autofarm.version import ARCHER_VERSION
from autofarm.winapi import ElevationMismatch, WinApi, WindowBindError

CONFIG_FILE = 'gui_config.json'
MAX_LOG_LINES = 500

# ---- 参数表：(键, 中文名)。时间参数 UI 上用毫秒，内部转秒 ----
PAIR_PARAMS = [                      # (min, max) 区间参数，UI 用毫秒
    ('jump_hold_secs', '跳跃键按住'),
    ('jump_rise_secs', '腾空等待'),
    ('key_hold_secs', '攻击键按住'),
    ('attack_gap_secs', '攻击间隔'),
    ('move_secs', '移动时长'),
    ('settle_secs', '停稳等待'),
    ('switch_gap_secs', '换边间隔'),
    ('idle_secs', '发呆时长'),
    ('potion_pause_secs', '喝药后停顿'),
    ('buff_pause_secs', 'Buff后停顿'),
]
SEC_PAIR_PARAMS = [                  # 秒级区间参数（数值大，不用毫秒）
    ('buff_every_secs', 'Buff间隔'),
]
INT_PARAMS = [                       # 整数参数
    ('attacks_per_side', '每边攻击次数'),
    ('potion_every', '每N下喝1药'),
    ('micro_move_every', '施法N次微调'),
]
PCT_PARAMS = [                       # 概率参数，UI 用 %
    ('extra_attack_prob', '多打1下%'),
    ('hop_prob', '纯跳一下%'),
    ('idle_prob', '轮间发呆%'),
]
KEY_PARAMS = [('attack_key', '攻击键'), ('jump_key', '跳跃键'),
              ('potion_key', '喝药键'), ('buff_key', 'Buff键')]


def _defaults():
    c = Config()
    d = {'window_title': c.window_title, 'plan': c.plan,
         'potion_enabled': c.potion_key is not None,
         'buff_enabled': c.buff_key is not None}
    for k, _ in PAIR_PARAMS:
        lo, hi = getattr(c, k)
        d[k + '_min'], d[k + '_max'] = str(int(lo * 1000)), str(int(hi * 1000))
    for k, _ in SEC_PAIR_PARAMS:
        lo, hi = getattr(c, k)
        d[k + '_min'], d[k + '_max'] = f'{lo:g}', f'{hi:g}'
    for k, _ in INT_PARAMS:
        d[k] = str(getattr(c, k))
    for k, _ in PCT_PARAMS:
        d[k] = str(int(round(getattr(c, k) * 100)))
    for k, _ in KEY_PARAMS:
        d[k] = getattr(c, k) or ''
    d['name_template'] = c.name_template or ''
    d['match_threshold'] = str(int(round(c.match_threshold * 100)))
    d['vision_interval'] = str(c.vision_interval)
    d['platform_left'] = '' if c.platform_left is None else str(c.platform_left)
    d['platform_right'] = '' if c.platform_right is None else str(c.platform_right)
    d['edge_margin'] = str(c.edge_margin)
    d['rescue_scale'] = str(c.rescue_scale)
    d['archer_profile'] = c.archer_profile
    d['archer_player_name'] = c.archer_player_name
    d['archer_name_template'] = c.archer_name_template
    d['archer_template_owner'] = c.archer_template_owner
    return d


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('冒险岛挂机助手')
        self.resizable(False, False)
        self.bot = None
        self.alive = False
        self._closing = False
        self.q = queue.Queue()
        self.vars = {}
        self._build()
        self._load()
        self.after(120, self._drain)
        self.protocol('WM_DELETE_WINDOW', self._on_close)

    # ---------- 界面 ----------

    def _var(self, name, value=''):
        self.vars[name] = tk.StringVar(value=str(value))
        return self.vars[name]

    def _row_pair(self, parent, row, label, key, unit='ms'):
        d = _defaults()
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky='e', padx=4, pady=2)
        f = ttk.Frame(parent)
        ttk.Entry(f, width=6, textvariable=self._var(key + '_min', d[key + '_min'])).pack(side='left')
        ttk.Label(f, text='~').pack(side='left')
        ttk.Entry(f, width=6, textvariable=self._var(key + '_max', d[key + '_max'])).pack(side='left')
        ttk.Label(f, text=unit).pack(side='left')
        f.grid(row=row, column=1, sticky='w', padx=4, pady=2)

    def _row_single(self, parent, row, label, key, unit=''):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky='e', padx=4, pady=2)
        f = ttk.Frame(parent)
        ttk.Entry(f, width=8, textvariable=self._var(key, _defaults()[key])).pack(side='left')
        if unit:
            ttk.Label(f, text=unit).pack(side='left')
        f.grid(row=row, column=1, sticky='w', padx=4, pady=2)

    def _build(self):
        top = ttk.LabelFrame(self, text='窗口与方案')
        top.grid(row=0, column=0, columnspan=2, sticky='we', padx=8, pady=(8, 2))
        ttk.Label(top, text='窗口标题').grid(row=0, column=0, sticky='e', padx=4, pady=2)
        ttk.Entry(top, width=24, textvariable=self._var('window_title', _defaults()['window_title'])).grid(
            row=0, column=1, sticky='w', padx=4)
        ttk.Label(top, text='方案').grid(row=1, column=0, sticky='e', padx=4, pady=2)
        self._var('plan', _defaults()['plan'])
        self._var('archer_player_name', _defaults()['archer_player_name'])
        self._var('archer_name_template', _defaults()['archer_name_template'])
        self._var('archer_template_owner', _defaults()['archer_template_owner'])
        ttk.Combobox(top, width=22, state='readonly', textvariable=self.vars['plan'],
                     values=list(plans.PLANS)).grid(row=1, column=1, sticky='w', padx=4)
        ttk.Button(top, text='载入绳边射手模板', command=self.on_archer_preset).grid(
            row=1, column=2, padx=4)
        ttk.Label(top, text='射手模板').grid(row=2, column=0, sticky='e', padx=4)
        ttk.Entry(top, width=48, textvariable=self._var('archer_profile', _defaults()['archer_profile'])).grid(
            row=2, column=1, columnspan=2, sticky='w', padx=4, pady=2)

        keys = ttk.LabelFrame(self, text='按键（小写键名，如 shift/alt/home/end/a）')
        keys.grid(row=1, column=0, sticky='we', padx=8, pady=2)
        for i, (k, label) in enumerate(KEY_PARAMS):
            ttk.Label(keys, text=label).grid(row=i, column=0, sticky='e', padx=4, pady=2)
            ttk.Entry(keys, width=10, textvariable=self._var(k, _defaults()[k])).grid(
                row=i, column=1, sticky='w', padx=4)
        self.potion_on = tk.BooleanVar(value=_defaults()['potion_enabled'])
        ttk.Checkbutton(keys, text='启用喝药', variable=self.potion_on).grid(
            row=4, column=0, columnspan=2, sticky='w', padx=4)
        self.buff_on = tk.BooleanVar(value=_defaults()['buff_enabled'])
        ttk.Checkbutton(keys, text='启用定时Buff（每隔一段时间按一次Buff键）',
                        variable=self.buff_on).grid(
            row=5, column=0, columnspan=2, sticky='w', padx=4)

        rhythm = ttk.LabelFrame(self, text='节奏参数（毫秒，随机区间）')
        rhythm.grid(row=1, column=1, sticky='nw', padx=8, pady=2)
        r = 0
        self._row_single(rhythm, r, INT_PARAMS[0][1], INT_PARAMS[0][0], '下'); r += 1
        for k, label in PAIR_PARAMS[:4]:
            self._row_pair(rhythm, r, label, k); r += 1

        rhythm2 = ttk.LabelFrame(self, text='节奏参数（续）')
        rhythm2.grid(row=2, column=1, sticky='nw', padx=8, pady=2)
        r = 0
        for k, label in PAIR_PARAMS[4:]:
            self._row_pair(rhythm2, r, label, k); r += 1

        rnd = ttk.LabelFrame(self, text='random_jump 随机概率（%）')
        rnd.grid(row=2, column=0, sticky='nw', padx=8, pady=2)
        r = 0
        for k, label in PCT_PARAMS:
            self._row_single(rnd, r, label, k, '%'); r += 1

        extra = ttk.LabelFrame(self, text='喝药 / Buff / 站桩方案')
        extra.grid(row=3, column=0, sticky='nw', padx=8, pady=2)
        self._row_single(extra, 0, INT_PARAMS[1][1], INT_PARAMS[1][0], '下')
        self._row_single(extra, 1, INT_PARAMS[2][1], INT_PARAMS[2][0], '次')
        self._row_pair(extra, 2, SEC_PAIR_PARAMS[0][1], SEC_PAIR_PARAMS[0][0], '秒')

        vis = ttk.LabelFrame(self, text='截图判断（名字牌找主角，防掉下去）')
        vis.grid(row=3, column=1, sticky='nw', padx=8, pady=2)
        self.vision_on = tk.BooleanVar(value=True)
        ttk.Checkbutton(vis, text='启用（关掉退回纯计时）', variable=self.vision_on).grid(
            row=0, column=0, columnspan=2, sticky='w', padx=4, pady=2)
        ttk.Label(vis, text='名字模板').grid(row=1, column=0, sticky='e', padx=4, pady=2)
        ttk.Entry(vis, width=22,
                  textvariable=self._var('name_template', _defaults()['name_template'])).grid(
            row=1, column=1, sticky='w', padx=4)
        self._row_single(vis, 2, '匹配阈值', 'match_threshold', '%')
        self._row_single(vis, 3, '截图间隔', 'vision_interval', '秒')
        self._row_single(vis, 4, '平台左边界', 'platform_left', 'px')
        self._row_single(vis, 5, '平台右边界', 'platform_right', 'px')
        self._row_single(vis, 6, '边缘留白', 'edge_margin', 'px')
        self._row_single(vis, 7, '贴边挪动倍数', 'rescue_scale', '倍')
        self.btn_calib = ttk.Button(vis, text='截屏标定（框名字牌）', command=self.on_calibrate)
        self.btn_calib.grid(row=8, column=0, sticky='w', padx=4, pady=4)
        self.btn_test = ttk.Button(vis, text='试一下识别', command=self.on_test_vision)
        self.btn_test.grid(row=8, column=1, sticky='w', padx=4, pady=4)

        ctrl = ttk.Frame(self)
        ctrl.grid(row=4, column=0, columnspan=2, sticky='we', padx=8, pady=6)
        self.btn_start = ttk.Button(ctrl, text='启动（挂后台等F12）', command=self.on_start)
        self.btn_start.pack(side='left', padx=4)
        self.btn_pause = ttk.Button(ctrl, text='开始挂机', command=self.on_pause, state='disabled')
        self.btn_pause.pack(side='left', padx=4)
        self.btn_stop = ttk.Button(ctrl, text='停止', command=self.on_stop, state='disabled')
        self.btn_stop.pack(side='left', padx=4)
        self.status = tk.StringVar(value='未运行')
        ttk.Label(ctrl, textvariable=self.status, foreground='#0a7').pack(side='left', padx=12)

        logf = ttk.LabelFrame(self, text='日志')
        logf.grid(row=5, column=0, columnspan=2, sticky='we', padx=8, pady=(0, 8))
        self.log_text = tk.Text(logf, height=12, width=72, state='disabled', font=('Consolas', 9))
        self.log_text.pack(side='left', fill='both', expand=True)
        sb = ttk.Scrollbar(logf, command=self.log_text.yview)
        sb.pack(side='right', fill='y')
        self.log_text['yscrollcommand'] = sb.set
        self._lines = 0

    # ---------- 配置持久化 ----------

    def _snapshot(self):
        return ({k: v.get() for k, v in self.vars.items()}
                | {'potion_enabled': self.potion_on.get(),
                   'buff_enabled': self.buff_on.get(),
                   'vision_enabled': self.vision_on.get()})

    def _save(self):
        try:
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(self._snapshot(), f, ensure_ascii=False, indent=1)
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
        for k, v in data.items():
            if k == 'potion_enabled':
                self.potion_on.set(bool(v))
            elif k == 'buff_enabled':
                self.buff_on.set(bool(v))
            elif k == 'vision_enabled':
                self.vision_on.set(bool(v))
            elif k in self.vars:
                self.vars[k].set(str(v))

    # ---------- 构建 Config ----------

    def _build_config(self):
        g = lambda k: self.vars[k].get().strip()
        ms = lambda k: (float(g(k + '_min')) / 1000.0, float(g(k + '_max')) / 1000.0)
        sec = lambda k: (float(g(k + '_min')), float(g(k + '_max')))
        opt_int = lambda k: (int(g(k)) if g(k) else None)
        if float(g('jump_rise_secs_min')) > float(g('jump_rise_secs_max')):
            raise ValueError('腾空等待最小值不能大于最大值')
        if float(g('buff_every_secs_min')) > float(g('buff_every_secs_max')):
            raise ValueError('Buff间隔最小值不能大于最大值')
        potion = g('potion_key').lower() if self.potion_on.get() else ''
        buff = g('buff_key').lower() if self.buff_on.get() else ''
        if g('plan') == 'rope_archer' and g('archer_name_template'):
            from autofarm.custom_template import check_owner
            check_owner(g('archer_player_name'), g('archer_template_owner'))
        return Config(
            window_title=g('window_title') or '冒险岛怀旧服',
            plan=g('plan'),
            attack_key=g('attack_key').lower() or 'shift',
            jump_key=g('jump_key').lower() or 'alt',
            potion_key=potion or None,
            buff_key=buff or None,
            attacks_per_side=int(g('attacks_per_side')),
            jump_hold_secs=ms('jump_hold_secs'),
            jump_rise_secs=ms('jump_rise_secs'),
            key_hold_secs=ms('key_hold_secs'),
            attack_gap_secs=ms('attack_gap_secs'),
            move_secs=ms('move_secs'),
            settle_secs=ms('settle_secs'),
            switch_gap_secs=ms('switch_gap_secs'),
            potion_every=int(g('potion_every')),
            potion_pause_secs=ms('potion_pause_secs'),
            buff_every_secs=sec('buff_every_secs'),
            buff_pause_secs=ms('buff_pause_secs'),
            extra_attack_prob=int(g('extra_attack_prob')) / 100.0,
            hop_prob=int(g('hop_prob')) / 100.0,
            idle_prob=int(g('idle_prob')) / 100.0,
            idle_secs=ms('idle_secs'),
            micro_move_every=int(g('micro_move_every')),
            vision_enabled=self.vision_on.get(),
            name_template=g('name_template'),
            match_threshold=int(g('match_threshold')) / 100.0,
            vision_interval=float(g('vision_interval')),
            platform_left=opt_int('platform_left'),
            platform_right=opt_int('platform_right'),
            edge_margin=int(g('edge_margin')),
            rescue_scale=float(g('rescue_scale')),
            archer_profile=g('archer_profile'),
            archer_player_name=g('archer_player_name'),
            archer_name_template=g('archer_name_template'),
            archer_template_owner=g('archer_template_owner'),
        )

    def on_archer_preset(self):
        if self.alive:
            return
        self.vars['plan'].set('rope_archer')
        self.vars['archer_profile'].set(Config().archer_profile)
        self.vars['attack_key'].set('shift')
        self.vision_on.set(True)
        self.buff_on.set(False)
        self.potion_on.set(False)
        self._save()
        self._log('[模板] 已载入绳边射手：使用独立模板里的边界和节奏；'
                  '长按 Shift，无怪松开，持续移动回位。先点「试一下识别」。')

    # ---------- 截图标定 ----------

    def on_calibrate(self):
        """截屏标定：先框名字牌，再框平台范围。只截图，不发任何按键。"""
        if self.vars['plan'].get() == 'rope_archer':
            messagebox.showinfo('独立射手模板', '绳边射手已按参考图标定，使用上方射手模板路径。\n'
                                '边界和节奏在 profile.json 中；此按钮仅标定旧方案。\n'
                                '请点「试一下识别」检查当前画面。')
            return
        try:
            cfg = self._build_config()
        except (ValueError, TypeError) as e:
            messagebox.showerror('参数错误', f'参数有误：{e}')
            return
        self._countdown(3, cfg)

    def _countdown(self, left, cfg, callback=None):
        if left > 0:
            self.status.set(f'{left} 秒后截图，请切回游戏窗口…')
            self.after(1000, lambda: self._countdown(left - 1, cfg, callback))
            return
        (callback or self._do_calibrate)(cfg)

    def _grab(self, cfg):
        """抓一张客户区截图；窗口找不到或不在前台就报错。"""
        api = WinApi()
        hwnd, _ = api.find_window(cfg.window_title)
        if api.get_foreground() != hwnd:
            raise RuntimeError('截图时游戏不在前台，请切回游戏再点一次')
        return V.capture(api.client_rect(hwnd))

    def _select(self, frame, tip):
        """弹窗让用户框选，返回 (x,y,w,h)；取消返回 None。"""
        roi = cv2.selectROI(tip, frame, False, False)
        cv2.destroyAllWindows()
        return None if roi[2] == 0 or roi[3] == 0 else tuple(int(v) for v in roi)

    def _do_calibrate(self, cfg):
        try:
            frame = self._grab(cfg)
        except Exception as e:
            messagebox.showerror('截图失败', str(e))
            self.status.set('未运行')
            return
        self._log(f'[标定] 客户区截图 {frame.shape[1]}x{frame.shape[0]}')
        roi = self._select(frame, '框选你的名字牌文字（如 CatApril）——回车确认 / Esc 取消')
        if roi is None:
            self._log('[标定] 已取消，没有改动')
            self.status.set('未运行')
            return
        x, y, w, h = roi
        folder = V.program_dir() / 'assets'
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f'name_{time.time_ns()}.png'
        cv2.imencode('.png', frame[y:y + h, x:x + w])[1].tofile(str(path))
        rel = path.relative_to(V.program_dir()).as_posix()
        self.vars['name_template'].set(rel)
        self._log(f'[标定] 名字牌模板 = {rel}（{w}x{h}）')
        plat = self._select(frame, '框选平台可站范围（木板左右两端）——回车确认 / Esc 跳过')
        if plat is not None:
            px, _, pw, _ = plat
            self.vars['platform_left'].set(str(int(px)))
            self.vars['platform_right'].set(str(int(px + pw)))
            self._log(f'[标定] 平台范围 = {int(px)} ~ {int(px + pw)} px')
        self._save()
        self.status.set('未运行')
        self._log('[标定] 已保存到 gui_config.json，点「试一下识别」可以校验')

    def on_test_vision(self):
        """截一张图试识别，把圈好的图存到 captures/，不发任何按键。"""
        try:
            cfg = self._build_config()
        except (ValueError, TypeError) as e:
            messagebox.showerror('参数错误', f'参数有误：{e}')
            return
        self._countdown(3, cfg, self._do_test_vision)

    def _do_test_vision(self, cfg):
        self.status.set('未运行')
        if cfg.plan == 'rope_archer':
            try:
                from autofarm.rope_archer import RopeScene
                scene = RopeScene(cfg.archer_profile, cfg.archer_player_name,
                                  cfg.archer_name_template, cfg.archer_template_owner)
                frame = self._grab(cfg)
                observation = scene.observe(frame)
                folder = V.program_dir() / 'captures'
                folder.mkdir(parents=True, exist_ok=True)
                cv2.imencode('.png', frame)[1].tofile(str(folder / 'rope_archer_raw.png'))
                out = folder / 'rope_archer_test.png'
                cv2.imencode('.png', scene.annotate(frame, observation))[1].tofile(str(out))
                text = observation.reason or (f'右侧有猴子（{observation.monkey_source}）：回到内侧并站稳后长按 Shift' if observation.monkey
                                              else '右侧无猴子：不攻击')
                if observation.player:
                    text += f'；{observation.player_source}，人物匹配 {observation.player.score:.3f}'
                self._log(f'[射手试识别 v{ARCHER_VERSION}] {text}；预览：{out}')
                messagebox.showinfo('射手识别结果', f'{text}\n\n预览：{out}\n'
                                    '红线=平台边界；橙线=主动右移上限；蓝线=回位目标。未发送按键。')
            except Exception as e:
                messagebox.showerror('射手试识别失败', str(e))
            return
        if not cfg.name_template:
            messagebox.showwarning('还没标定', '先点「截屏标定」框一下名字牌')
            return
        try:
            frame = self._grab(cfg)
            finder = V.NameFinder(cfg.name_template, cfg.match_threshold)
            hit = finder.find(frame)
            left = 0 if cfg.platform_left is None else cfg.platform_left
            right = frame.shape[1] - 1 if cfg.platform_right is None else cfg.platform_right
            direction, rescue = V.choose_direction(
                hit.x if hit else None, left, right, cfg.edge_margin)
        except Exception as e:
            messagebox.showerror('试识别失败', str(e))
            return
        folder = V.program_dir() / 'captures'
        folder.mkdir(parents=True, exist_ok=True)
        if hit is None:
            text = f'not found (threshold {cfg.match_threshold:.2f})'
            message = (f'没找到名字牌（阈值 {cfg.match_threshold:.2f}）。'
                       '可以调低阈值，或者重新标定模板。')
        else:
            lane = '贴边回中间' if rescue else '常规'
            side = '在左半边' if direction == 'right' else '在右半边'
            text = f'x={hit.x:.0f} score={hit.score:.2f} -> {direction}'
            message = (f'x={hit.x:.0f}，得分 {hit.score:.2f}：{side} → 先打'
                       f'{"右" if direction == "right" else "左"}（{lane}）')
        out = folder / 'vision_test.png'
        cv2.imencode('.png', V.annotate(frame, hit, text))[1].tofile(str(out))
        self._log(f'[试识别] {message}')
        self._log(f'[试识别] 圈好的图：{out}')
        messagebox.showinfo('识别结果', f'{message}\n\n圈好的图已存到 captures/vision_test.png')

    # ---------- 动作 ----------

    def _log(self, msg):
        self.log_text['state'] = 'normal'
        self.log_text.insert('end', msg + '\n')
        self._lines += 1
        if self._lines > MAX_LOG_LINES:
            self.log_text.delete('1.0', '2.0')
            self._lines -= 1
        self.log_text.see('end')
        self.log_text['state'] = 'disabled'

    def on_start(self):
        if self.alive:
            return
        try:
            cfg = self._build_config()
        except (ValueError, TypeError) as e:
            messagebox.showerror('参数错误', f'参数有误：{e}')
            return
        self._save()
        self.alive = True
        self.btn_start['state'] = 'disabled'
        self.btn_pause['state'] = 'normal'
        self.btn_stop['state'] = 'normal'
        self.btn_calib['state'] = 'disabled'
        self.btn_test['state'] = 'disabled'
        self.btn_pause['text'] = '开始挂机'
        self.status.set('启动中…')
        threading.Thread(target=self._worker, args=(cfg,), daemon=True).start()

    def on_pause(self):
        if self.bot is None:
            return
        self.bot.paused = not self.bot.paused
        if self.bot.paused:
            self.q.put(('log', '[暂停] 手动暂停'))
            self.btn_pause['text'] = '开始挂机'
        else:
            self.q.put(('log', '[运行] 手动开始'))
            self.btn_pause['text'] = '暂停'

    def on_stop(self):
        if self.bot is not None:
            self.bot.quitting = True
            self.status.set('正在停止…')

    def _worker(self, cfg):
        q = self.q
        bot = None
        try:
            bot = Bot(cfg.window_title, log=lambda m: q.put(('log', m)))
            self.bot = bot
            q.put(('log', f'绑定窗口：{bot.window_name}'))
            q.put(('log', f'方案：{cfg.plan}'))
            q.put(('log', '已暂停：按 F12 或点「开始挂机」开始；F11 或点「停止」退出。'))
            plans.PLANS[cfg.plan](bot, cfg)
        except BotStopped:
            q.put(('log', '[退出] 收到停止请求。'))
        except ElevationMismatch as e:
            q.put(('log', f'[错误] {e}'))
            q.put(('elevation', None))
        except WindowBindError as e:
            q.put(('log', f'[错误] {e}'))
        except Exception as e:                       # 兜底：任何异常都显示在日志里
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
            self.status.set('已暂停（F12/按钮继续）' if self.bot.paused else '挂机运行中…')
        self.after(120, self._drain)

    def _on_done(self):
        self.alive = False
        self.bot = None
        self.status.set('已停止')
        self.btn_start['state'] = 'normal'
        self.btn_pause['state'] = 'disabled'
        self.btn_pause['text'] = '开始挂机'
        self.btn_stop['state'] = 'disabled'
        self.btn_calib['state'] = 'normal'
        self.btn_test['state'] = 'normal'

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
        """用 runas 拉起自己（会弹 UAC）；取消或失败就留在当前界面。"""
        args = ' '.join(f'"{a}"' for a in sys.argv[1:]) or None
        rc = ctypes.windll.shell32.ShellExecuteW(None, 'runas', sys.executable, args, None, 1)
        if rc <= 32:                                   # <=32 表示失败或被用户取消
            messagebox.showwarning('提权未完成',
                                   '以管理员身份重启被取消或失败，'
                                   '请手动右键程序选择「以管理员身份运行」。')
            return
        self._on_close()


def main():
    if getattr(sys, 'frozen', False):                # exe 模式下配置文件放 exe 旁边
        os.chdir(os.path.dirname(sys.executable))
    try:
        ctypes.windll.user32.SetProcessDPIAware()    # 截图按物理像素，别被缩放改写坐标
    except Exception:
        pass
    App().mainloop()


if __name__ == '__main__':
    main()
