"""One window for random jump, rope archer, and independent Buff timers."""
import ctypes
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import sys
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image
import cv2

import farm_gui
from rope_archer_gui import ArcherApp
from buff_gui import App as BuffApp, DEFAULT_SLOTS
from autofarm.buffs import parse_slots
from autofarm.winapi import VK_CODES

VERSION = '2.2.0'
CONFIG_FILE = 'unified_config.json'
MODES = {'random_jump': '随机跳攻 · 野猪领地1',
         'rope_archer': '绳边射手 · 猴子沼泽3',
         'platform_guard': '自定义守台 · 用户标定',
         'buff_only': '仅定时 Buff · 手动操作时补技能'}


def read_settings(folder):
    """Migrate copies of old settings; never rewrite the original files."""
    def read(name):
        path = folder / name
        if not path.exists():
            return {}
        data = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(data, dict):
            raise ValueError(f'{name} 配置格式错误')
        return data
    if (folder / CONFIG_FILE).exists():
        return read(CONFIG_FILE)
    general, archer, buffs = read('gui_config.json'), read('rope_archer_config.json'), read('buff_gui_config.json')
    data = dict(general)
    data.update({k: v for k, v in archer.items() if k.startswith('archer_')})
    data['window_title'] = archer.get('window_title', general.get('window_title', '冒险岛怀旧服'))
    data['plan'] = 'rope_archer' if archer else 'random_jump'
    if 'slots' in buffs:
        data['buff_slots'] = buffs['slots']
    elif general.get('buff_enabled'):
        data['buff_slots'] = [dict(key=general.get('buff_key', 'home'),
                                  lo=general.get('buff_every_secs_min', '60'),
                                  hi=general.get('buff_every_secs_max', '90'),
                                  pmin=str(float(general.get('buff_pause_secs_min', 300)) / 1000),
                                  pmax=str(float(general.get('buff_pause_secs_max', 600)) / 1000))]
    # Import keys/intervals, but leave the new shared module opt-in on migration.
    data['buff_enabled'] = False
    return data


class UnifiedApp(ArcherApp):
    _add_slot_row = BuffApp._add_slot_row
    _remove_slot = BuffApp._remove_slot
    _slots = BuffApp._slots

    def __init__(self):
        farm_gui.CONFIG_FILE = CONFIG_FILE
        super().__init__()

    def entry(self, parent, label, key, row, width=46):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky='w', padx=8, pady=5)
        ttk.Entry(parent, textvariable=self.vars[key], width=width).grid(row=row, column=1, sticky='w', padx=8, pady=5)

    def pair(self, parent, label, key, row):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky='w', padx=8, pady=3)
        f = ttk.Frame(parent)
        f.grid(row=row, column=1, sticky='w')
        for suffix in ('min', 'max'):
            ttk.Entry(f, textvariable=self.vars[key + '_' + suffix], width=8).pack(side='left', padx=5)

    def _build(self):
        for key, value in farm_gui._defaults().items():
            self._var(key, value)
        for key, value in dict(guard_profile='templates/platform_guard_example/profile.json', guard_player_name='', guard_direction='right',
                               guard_attack_key='shift', guard_attack_range='450', buff_hold_ms='120').items():
            self._var(key, value)
        self.buff_start = tk.BooleanVar(value=True)
        self.potion_on = tk.BooleanVar(value=False)
        self.buff_on = tk.BooleanVar(value=False)
        self.vision_on = tk.BooleanVar(value=True)
        top = ttk.LabelFrame(self, text='冒险岛统一助手')
        top.pack(fill='x', padx=12, pady=8)
        self.entry(top, '游戏窗口标题', 'window_title', 0)
        self.mode_label = tk.StringVar()
        self.mode_box = ttk.Combobox(top, textvariable=self.mode_label, values=list(MODES.values()), state='readonly', width=44)
        self.mode_box.grid(row=1, column=1, padx=8, pady=5)
        ttk.Label(top, text='运行模式').grid(row=1, column=0, padx=8, sticky='w')
        self.mode_box.bind('<<ComboboxSelected>>', self._select_mode)
        self.tabs = ttk.Notebook(self)
        self.tabs.pack(fill='both', expand=True, padx=12)
        self.random_tab = ttk.Frame(self.tabs)
        self.archer_tab = ttk.Frame(self.tabs)
        self.buff_tab = ttk.Frame(self.tabs)
        self.guard_tab = ttk.Frame(self.tabs)
        for tab, name in [(self.random_tab, '随机跳攻'), (self.archer_tab, '绳边射手'),
                          (self.guard_tab, '自定义守台'), (self.buff_tab, '定时 Buff')]:
            self.tabs.add(tab, text=name)
        self._random_panel()
        self._archer_panel()
        self._guard_panel()
        self._buff_panel()
        ttk.Label(self, text='选择运行模式后启动；F12 开始 / 暂停，F11 停止。切出游戏自动挂起。', padding=8).pack(anchor='w')
        controls = ttk.Frame(self)
        controls.pack(fill='x', padx=12)
        for attr, label, command, state in [
            ('btn_start', '启动（等待 F12）', self.on_start, 'normal'),
            ('btn_pause', '开始挂机', self.on_pause, 'disabled'),
            ('btn_stop', '停止', self.on_stop, 'disabled')]:
            button = ttk.Button(controls, text=label, command=command, state=state)
            button.pack(side='left', padx=3)
            setattr(self, attr, button)
        self.status = tk.StringVar(value='未运行')
        ttk.Label(controls, textvariable=self.status, foreground='#087', padding=8).pack(side='left')
        self.log_text = tk.Text(self, height=8, width=88, state='disabled', font=('Consolas', 9))
        self.log_text.pack(fill='x', padx=12, pady=10)
        self._lines = 0

    def _random_panel(self):
        panel = self.random_tab
        ttk.Label(panel, text='野猪领地1：每次基础攻击按左右概率分配；左右移动仍成对执行。', padding=8).pack(anchor='w')
        columns = ttk.Frame(panel)
        columns.pack(fill='x')
        keys = ttk.LabelFrame(columns, text='按键与概率')
        keys.pack(side='left', fill='both', padx=5)
        for row, (label, key) in enumerate([('攻击键', 'attack_key'), ('跳跃键', 'jump_key'),
                ('追加攻击概率 %', 'extra_attack_prob'), ('纯跳概率 %', 'hop_prob'), ('停顿概率 %', 'idle_prob'),
                ('喝药键', 'potion_key'), ('每多少次攻击喝药', 'potion_every')]):
            self.entry(keys, label, key, row, 8)
        self.entry(keys, '向右攻击概率 %', 'right_attack_prob', 7, 8)
        self.left_probability = tk.StringVar()
        ttk.Label(keys, text='向左攻击概率 %').grid(row=8, column=0, sticky='w', padx=8, pady=5)
        ttk.Entry(keys, textvariable=self.left_probability, state='readonly', width=8).grid(row=8, column=1, sticky='w', padx=8)
        self.vars['right_attack_prob'].trace_add('write', self._update_left_probability)
        self._update_left_probability()
        ttk.Checkbutton(keys, text='启用喝药（随机跳攻）', variable=self.potion_on).grid(row=9, columnspan=2, pady=5)
        timing = ttk.LabelFrame(columns, text='节奏：最小 / 最大（毫秒）')
        timing.pack(side='left', fill='both', padx=5)
        for row, (key, label) in enumerate(farm_gui.PAIR_PARAMS):
            if key != 'buff_pause_secs':
                self.pair(timing, label, key, row)

    def _update_left_probability(self, *args):
        try:
            right = float(self.vars['right_attack_prob'].get())
            if not math.isfinite(right) or not 0 <= right <= 100:
                raise ValueError
            self.left_probability.set(f'{100 - right:g}')
        except ValueError:
            self.left_probability.set('—')

    def _archer_panel(self):
        panel = self.archer_tab
        settings = ttk.Frame(panel)
        settings.pack(fill='x')
        self.entry(settings, '角色名（可选）', 'archer_player_name', 0)
        self.entry(settings, '平台模板路径', 'archer_profile', 1)
        ttk.Label(panel, text='猴子沼泽3：确认朝右后按 Shift；先清怪再回位；人物高度不拦截。\n名字留空用原图片；填写名字用本地 OCR；粘贴图片后优先用该图片。', padding=8).pack(anchor='w')
        custom = ttk.LabelFrame(panel, text='名字图片模板')
        custom.pack(fill='x', padx=8)
        buttons = ttk.Frame(custom)
        buttons.pack(fill='x', padx=8, pady=4)
        ttk.Button(buttons, text='粘贴图片', command=self._paste_name_template).pack(side='left')
        ttk.Button(buttons, text='清除模板', command=self._clear_name_template).pack(side='left', padx=6)
        ttk.Label(buttons, text='复制截图 → 粘贴 → 框选名字一行').pack(side='left')
        self.template_canvas = tk.Canvas(custom, width=560, height=58, background='#20242c', takefocus=True)
        self.template_canvas.pack(padx=8, pady=3)
        self.template_canvas.bind('<Button-1>', lambda event: self.template_canvas.focus_set())
        self.template_status = tk.StringVar()
        ttk.Label(custom, textvariable=self.template_status, wraplength=620).pack(anchor='w', padx=8, pady=4)
        self._template_loading = False
        self.vars['archer_player_name'].trace_add('write', lambda *args: self._refresh_name_template())
        self.bind('<Control-v>', self._paste_shortcut, add='+')
        checks = ttk.Frame(panel)
        checks.pack(fill='x', padx=8, pady=8)
        self.btn_test = ttk.Button(checks, text='试一下射手识别（3秒后截图）', command=self.on_test_vision)
        self.btn_test.pack(side='left')
        self.btn_calib = ttk.Button(checks, text='模板说明', command=lambda: messagebox.showinfo('模板说明', '请先选择绳边射手模式并测试识别。\n平台范围与识别阈值位于模板路径中的 profile.json。', parent=self))
        self.btn_calib.pack(side='left', padx=6)

    def _buff_panel(self):
        panel = self.buff_tab
        ttk.Checkbutton(panel, text='挂机时启用定时 Buff（开关立即应用）', variable=self.buff_on,
                        command=self._apply_buffs).pack(anchor='w', padx=8, pady=5)
        first = ttk.Frame(panel)
        first.pack(fill='x', padx=8)
        ttk.Checkbutton(first, text='首次开始时先补一次', variable=self.buff_start).pack(side='left')
        ttk.Label(first, text='按住毫秒').pack(side='left', padx=6)
        ttk.Entry(first, textvariable=self.vars['buff_hold_ms'], width=6).pack(side='left')
        ttk.Label(panel, text='仅 Buff 模式总是启用；每行独立计时。修改槽位后点「应用」。\n开始后需 F12 运行且游戏在前台；射手 / 守台会先松开攻击，等 150ms 再施放。', padding=8).pack(anchor='w')
        ttk.Label(panel, text='按键          间隔最小秒  间隔最大秒  停顿最小秒  停顿最大秒', padding=8).pack(anchor='w')
        canvas = tk.Canvas(panel, height=120, highlightthickness=0)
        canvas.pack(fill='both', expand=True, padx=8)
        self.slots_frame = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=self.slots_frame, anchor='nw')
        self.slots_frame.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        scrollbar = ttk.Scrollbar(canvas, orient='vertical', command=canvas.yview)
        scrollbar.pack(side='right', fill='y')
        canvas.configure(yscrollcommand=scrollbar.set)
        self.slot_rows = []
        for slot in DEFAULT_SLOTS:
            self._add_slot_row(**slot)
        controls = ttk.Frame(panel)
        controls.pack(fill='x', padx=8, pady=6)
        ttk.Button(controls, text='+ 添加 Buff', command=self._add_slot_row).pack(side='left')
        ttk.Button(controls, text='应用 Buff 设置', command=self._apply_buffs).pack(side='left', padx=5)
        ttk.Button(controls, text='补一次（回游戏后执行）', command=self._buff_now).pack(side='left')
        self.buff_status = tk.StringVar(value='未运行；勾选启用后启动，或选择仅 Buff 模式')
        ttk.Label(panel, textvariable=self.buff_status, wraplength=610, foreground='#087', padding=8).pack(anchor='w')

    def _buff_options(self):
        plan = self.vars['plan'].get()
        forbidden = () if plan == 'buff_only' else (self.vars['attack_key'].get().lower(),
            self.vars['jump_key'].get().lower(), self.vars['guard_attack_key'].get().lower(), 'shift')
        enabled = self.buff_on.get() or plan == 'buff_only'
        slots = parse_slots(self._slots(), forbidden) if enabled else ()
        if enabled and not slots:
            raise ValueError('请添加至少一个有效的 Buff 槽位')
        hold = float(self.vars['buff_hold_ms'].get()) / 1000
        if not math.isfinite(hold) or not .03 <= hold <= 1:
            raise ValueError('Buff 按住时间必须在 30~1000 毫秒之间')
        return slots, hold

    def _apply_buffs(self):
        try:
            slots, hold = self._buff_options()
            self._save()
            if self.alive and self.bot is not None:
                self.bot.buff_commands.put(('hold', hold))
                self.bot.buff_commands.put(('replace', slots))
                self.buff_status.set('设置已提交；回到游戏并运行后应用')
            else:
                self.buff_status.set('已保存：' + (', '.join(s.key for s in slots) if slots else 'Buff 关闭'))
            return True
        except (ValueError, OSError) as error:
            messagebox.showerror('Buff 设置未应用', str(error), parent=self)
            return False

    def _buff_now(self):
        try:
            slots, _ = self._buff_options()
            if not slots:
                raise ValueError('请先勾选启用 Buff')
            if not self.alive or self.bot is None:
                raise ValueError('请先启动任务，再点补一次；回到游戏按 F12 运行')
            if not self._apply_buffs():
                return
            self.bot.buff_commands.put(('now', None))
            self.buff_status.set('已安排补一次，等待游戏前台和运行状态')
        except ValueError as error:
            messagebox.showinfo('Buff', str(error), parent=self)

    def _drain(self):
        super()._drain()
        if not self._closing and self.bot is not None:
            state = '已暂停；' if self.bot.paused else ''
            self.buff_status.set(state + self.bot.buff_status)

    def _guard_panel(self):
        panel = self.guard_tab
        settings = ttk.Frame(panel)
        settings.pack(fill='x')
        for row, (label, key) in enumerate([('方案路径', 'guard_profile'), ('角色名（可选 OCR）', 'guard_player_name'),
                                            ('攻击键', 'guard_attack_key'), ('水平射程（像素）', 'guard_attack_range')]):
            self.entry(settings, label, key, row, 42 if row < 2 else 10)
        ttk.Label(settings, text='攻击方向').grid(row=4, column=0, sticky='w', padx=8, pady=5)
        ttk.Combobox(settings, textvariable=self.vars['guard_direction'], values=('right', 'left', 'both'),
                     state='readonly', width=10).grid(row=4, column=1, sticky='w', padx=8)
        ttk.Label(panel, text='right=只向右；left=只向左；both=两侧，优先清理当前方向。\n框选参照物、名字、平台边界、停靠范围、观察区域及怪物。\n名字留空使用标定图片；高度不拦截动作。', padding=8).pack(anchor='w')
        actions = ttk.Frame(panel)
        actions.pack(fill='x', padx=8)
        for label, command in [('截图标定（3秒）', self._guard_capture), ('粘贴画面标定', self._guard_clipboard),
                               ('载入方案', self._guard_load), ('编辑标定', self._guard_edit)]:
            ttk.Button(actions, text=label, command=command).pack(side='left', padx=2)
        extra = ttk.Frame(panel)
        extra.pack(fill='x', padx=8, pady=8)
        ttk.Button(extra, text='追加怪物图片（剪贴板）', command=self._guard_add_monster).pack(side='left')
        ttk.Button(extra, text='测试守台识别（3秒）', command=self._guard_test).pack(side='left', padx=5)
        self.guard_status = tk.StringVar(value='内置猴子沼泽3示例；更换地图请截图标定，可保存多个方案')
        ttk.Label(panel, textvariable=self.guard_status, wraplength=610, padding=8).pack(anchor='w')

    def _guard_idle(self):
        if self.alive:
            messagebox.showinfo('请先停止', '修改标定或测试识别前，请停止运行中的任务。', parent=self)
            return False
        return True

    def _guard_saved(self, path):
        resolved = Path(path).resolve()
        root = farm_gui.V.program_dir().resolve()
        value = resolved.relative_to(root).as_posix() if resolved.is_relative_to(root) else str(resolved)
        self.vars['guard_profile'].set(value)
        self._save()
        name = json.loads(resolved.read_text(encoding='utf-8')).get('name', resolved.parent.name)
        self.guard_status.set(f'已选择：{name}；先测试识别，再启动')

    def _guard_dialog(self, image, existing=None):
        from guard_template_gui import GuardCalibrationDialog
        GuardCalibrationDialog(self, image, self._guard_saved, self.vars['guard_player_name'].get(), existing)

    def _guard_capture(self):
        if self._guard_idle():
            cfg = farm_gui.Config(window_title=self.vars['window_title'].get())
            self._countdown(3, cfg, self._guard_capture_done)

    def _guard_capture_done(self, cfg):
        if not self._guard_idle():
            return
        try:
            frame = self._grab(cfg)
            self._guard_dialog(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        except Exception as error:
            messagebox.showerror('标定截图失败', str(error), parent=self)
        self.status.set('未运行')

    def _guard_clipboard(self):
        if self._guard_idle():
            try:
                from autofarm.custom_template import clipboard_image
                self._guard_dialog(clipboard_image())
            except (ValueError, OSError) as error:
                messagebox.showerror('粘贴失败', str(error), parent=self)

    def _guard_load(self):
        if not self._guard_idle():
            return
        path = filedialog.askopenfilename(title='选择守台方案', filetypes=[('守台方案', '*.json')])
        if path:
            try:
                from autofarm.platform_guard import GuardScene
                GuardScene(path)
                self._guard_saved(path)
            except (ValueError, OSError) as error:
                messagebox.showerror('方案无效', str(error), parent=self)

    def _guard_edit(self):
        if not self._guard_idle():
            return
        try:
            from autofarm.platform_guard import profile_asset, validate_profile
            path = farm_gui.V.asset_path(self.vars['guard_profile'].get())
            d = validate_profile(json.loads(path.read_text(encoding='utf-8')))
            with Image.open(profile_asset(path, d['reference_image'])) as image:
                self._guard_dialog(image.copy(), d | {'_profile_path': str(path)})
        except (ValueError, OSError, KeyError) as error:
            messagebox.showerror('无法编辑方案', str(error), parent=self)

    def _guard_add_monster(self):
        if not self._guard_idle():
            return
        try:
            from autofarm.custom_template import clipboard_image
            from guard_template_gui import MonsterCropDialog, append_monster
            path = self.vars['guard_profile'].get()
            if not path:
                raise ValueError('请先标定或载入方案')
            def saved(image):
                append_monster(path, image)
                self.guard_status.set('怪物模板已追加；将同时匹配原模板和新模板')
            MonsterCropDialog(self, clipboard_image(), saved)
        except (ValueError, OSError) as error:
            messagebox.showerror('无法追加怪物', str(error), parent=self)

    def _guard_test(self):
        if self._guard_idle():
            try:
                self._countdown(3, self._build_config(), self._guard_test_done)
            except ValueError as error:
                messagebox.showerror('配置无效', str(error), parent=self)

    def _guard_test_done(self, cfg):
        if not self._guard_idle():
            return
        try:
            from autofarm.platform_guard import GuardScene
            scene = GuardScene(cfg.guard_profile, cfg.guard_player_name, cfg.guard_attack_range)
            frame = self._grab(cfg)
            o = scene.observe(frame)
            output = farm_gui.V.program_dir() / 'captures' / 'platform_guard_test.png'
            output.parent.mkdir(exist_ok=True)
            cv2.imencode('.png', scene.annotate(frame, o))[1].tofile(str(output))
            result = o.reason or f'人物已找到；左侧{"有怪" if o.left else "无怪"}，右侧{"有怪" if o.right else "无怪"}'
            self.guard_status.set(result)
            self._log(f'[守台测试] {result}；预览：{output}')
            from guard_template_gui import show_preview
            show_preview(self, Image.fromarray(cv2.cvtColor(scene.annotate(frame, o), cv2.COLOR_BGR2RGB)),
                         f'{result}\n红线：平台边界；蓝线：期望位置；黄圈：怪物。已保存：{output}')
        except Exception as error:
            messagebox.showerror('守台测试失败', str(error), parent=self)
        self.status.set('未运行')

    def _select_mode(self, event=None):
        plan = next(k for k, label in MODES.items() if label == self.mode_label.get())
        self.vars['plan'].set(plan)
        self.tabs.select({'random_jump': self.random_tab, 'rope_archer': self.archer_tab,
                          'platform_guard': self.guard_tab, 'buff_only': self.buff_tab}[plan])

    def _load(self):
        self._template_loading = True
        try:
            data = read_settings(Path.cwd())
            for key, value in data.items():
                if key == 'buff_start_immediately':
                    self.buff_start.set(bool(value))
                elif key in ('buff_enabled', 'potion_enabled'):
                    (self.buff_on if key == 'buff_enabled' else self.potion_on).set(bool(value))
                elif key in self.vars:
                    self.vars[key].set(str(value))
            if 'buff_slots' in data:
                for row in list(self.slot_rows):
                    self._remove_slot(row['frame'])
                for slot in data['buff_slots']:
                    self._add_slot_row(**{k: slot.get(k, v) for k, v in DEFAULT_SLOTS[0].items()})
        except (OSError, ValueError, TypeError, AttributeError) as error:
            self._log(f'[配置] 加载失败，使用当前默认值：{error}')
        self.vision_on.set(True)
        plan = self.vars['plan'].get()
        self.mode_label.set(MODES.get(plan, MODES['random_jump']))
        self._select_mode()
        self._template_loading = False
        self._refresh_name_template()
        self.title(f'冒险岛统一助手 v{VERSION} · 自定义守台 / 随机跳攻 / 绳边射手 / Buff')
        self._log(f'[版本] 统一助手 v{VERSION}；配置：{Path(CONFIG_FILE).resolve()}')
        self._log(f'[程序] {Path(sys.executable if getattr(sys, "frozen", False) else __file__).resolve()}')

    def _snapshot(self):
        return super()._snapshot() | {'buff_slots': self._slots(), 'version': VERSION,
                                      'buff_start_immediately': self.buff_start.get()}

    def _save(self):
        target = Path(CONFIG_FILE)
        temp = target.with_suffix('.json.tmp')
        temp.write_text(json.dumps(self._snapshot(), ensure_ascii=False, indent=2), encoding='utf-8')
        os.replace(temp, target)

    def _build_config(self):
        cfg = farm_gui.App._build_config(self)
        if cfg.plan not in MODES:
            raise ValueError('请选择有效的运行模式')
        if not math.isfinite(cfg.right_attack_prob) or not 0 <= cfg.right_attack_prob <= 1:
            raise ValueError('向右攻击概率必须在 0~100% 之间，向左概率自动补足到 100%')
        for key in (cfg.attack_key, cfg.jump_key, cfg.potion_key):
            if key and (key not in VK_CODES or key in ('f11', 'f12')):
                raise ValueError(f'无效或保留按键：{key}')
        for key, _ in farm_gui.PAIR_PARAMS:
            pair = getattr(cfg, key)
            if not all(math.isfinite(v) for v in pair) or pair[0] < 0 or pair[0] > pair[1]:
                raise ValueError(f'{key} 的时间范围无效')
        if cfg.potion_every <= 0 or not all(0 <= v <= 1 for v in (cfg.extra_attack_prob, cfg.hop_prob, cfg.idle_prob)):
            raise ValueError('喝药频率必须大于零；概率必须在 0~100 之间')
        slots, hold = self._buff_options()
        attack = self.vars['guard_attack_key'].get().strip().lower()
        direction = self.vars['guard_direction'].get()
        reach = float(self.vars['guard_attack_range'].get())
        if attack not in VK_CODES or attack in ('left', 'right', 'up', 'down', 'f11', 'f12'):
            raise ValueError('自定义守台攻击键无效或与移动 / 热键冲突')
        if direction not in ('right', 'left', 'both') or not math.isfinite(reach) or not 50 <= reach <= 1000:
            raise ValueError('守台方向无效，或射程不在 50~1000 像素范围内')
        return replace(cfg, buff_key=None, buff_slots=slots, buff_hold_secs=hold,
                       buff_start_immediately=self.buff_start.get(), vision_enabled=True,
                       guard_profile=self.vars['guard_profile'].get(), guard_player_name=self.vars['guard_player_name'].get(),
                       guard_direction=direction, guard_attack_key=attack, guard_attack_range=reach)

    def on_start(self):
        if self.vars['plan'].get() == 'platform_guard' and not self.vars['guard_profile'].get():
            messagebox.showinfo('请先标定', '请先在自定义守台页标定或载入方案。', parent=self)
            return
        try:
            super().on_start()
        except OSError as error:
            messagebox.showerror('无法保存配置', str(error), parent=self)
        if self.alive:
            self.mode_box.configure(state='disabled')
            self._log('[Buff] ' + ('已启用，F12 运行后执行' if self.buff_on.get() or self.vars['plan'].get() == 'buff_only' else '未启用：可到定时 Buff 页勾选'))

    def _on_done(self):
        super()._on_done()
        self.mode_box.configure(state='readonly')

    def on_test_vision(self):
        if self.vars['plan'].get() != 'rope_archer':
            messagebox.showinfo('选择模式', '请先在顶部选择「绳边射手 · 猴子沼泽3」。', parent=self)
            return
        super().on_test_vision()


def main():
    os.chdir(Path(sys.executable).resolve().parent if getattr(sys, 'frozen', False) else Path(__file__).resolve().parent)
    ctypes.windll.user32.SetProcessDPIAware()
    UnifiedApp().mainloop()


if __name__ == '__main__':
    main()
