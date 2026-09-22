"""独立入口和配置，保留现有挂机方案的设置。"""
import os
from pathlib import Path
import farm_gui
from tkinter import ttk
from autofarm.version import ARCHER_VERSION

farm_gui.CONFIG_FILE = 'rope_archer_config.json'
OriginalApp = farm_gui.App


class ArcherApp(OriginalApp):
    def _build(self):
        # 独立入口只展示射手会使用的设置，沿用原有运行/停止/日志机制。
        d = farm_gui._defaults()
        for key, value in d.items():
            self._var(key, value)
        self.potion_on = farm_gui.tk.BooleanVar(value=False)
        self.buff_on = farm_gui.tk.BooleanVar(value=False)
        self.vision_on = farm_gui.tk.BooleanVar(value=True)
        box = ttk.LabelFrame(self, text='绳边射手 · 猴子沼泽地3')
        box.pack(fill='x', padx=12, pady=10)
        for row, (label, key) in enumerate([('窗口标题', 'window_title'), ('独立模板', 'archer_profile')]):
            ttk.Label(box, text=label).grid(row=row, column=0, padx=8, pady=5)
            ttk.Entry(box, textvariable=self.vars[key], width=52).grid(row=row, column=1, padx=8, pady=5)
        ttk.Label(box, text='右侧有猴子 → 长按 Shift；无猴子 → 松开\n'
                  '被击退 → 向平台内侧回位；钱币遮住身体 → 辅助识别头顶光圈',
                  justify='left').grid(row=2, column=0, columnspan=2, padx=8, pady=8, sticky='w')
        checks = ttk.Frame(self)
        checks.pack(fill='x', padx=12)
        self.btn_test = ttk.Button(checks, text='试一下识别（3秒后截图）', command=self.on_test_vision)
        self.btn_test.pack(side='left', padx=3)
        self.btn_calib = ttk.Button(checks, text='模板说明', command=self.on_calibrate)
        self.btn_calib.pack(side='left', padx=3)
        ttk.Label(self, text='先测试识别，再启动。切回游戏按 F12 开始/暂停，F11 停止。',
                  padding=10).pack(anchor='w')
        ctrl = ttk.Frame(self)
        ctrl.pack(fill='x', padx=12)
        self.btn_start = ttk.Button(ctrl, text='启动（挂后台等F12）', command=self.on_start)
        self.btn_start.pack(side='left', padx=3)
        self.btn_pause = ttk.Button(ctrl, text='开始挂机', command=self.on_pause, state='disabled')
        self.btn_pause.pack(side='left', padx=3)
        self.btn_stop = ttk.Button(ctrl, text='停止', command=self.on_stop, state='disabled')
        self.btn_stop.pack(side='left', padx=3)
        self.status = farm_gui.tk.StringVar(value='未运行')
        ttk.Label(self, textvariable=self.status, foreground='#0a7', padding=10).pack(anchor='w')
        self.log_text = farm_gui.tk.Text(self, height=11, width=76, state='disabled', font=('Consolas', 9))
        self.log_text.pack(padx=12, pady=(0, 12))
        self._lines = 0

    def _load(self):
        if Path(farm_gui.CONFIG_FILE).exists():
            super()._load()
        else:
            self.on_archer_preset()
        self.vars['plan'].set('rope_archer')
        self.vision_on.set(True)
        self.title(f'冒险岛 · 绳边射手 v{ARCHER_VERSION}（内侧站位 / 光圈识别）')
        self._log(f'[版本] 绳边射手 v{ARCHER_VERSION}：内侧站位 / 光圈识别')
        entry = farm_gui.sys.executable if getattr(farm_gui.sys, 'frozen', False) else __file__
        self._log(f'[程序] {Path(entry).resolve()}')


if __name__ == '__main__':
    os.chdir(Path(__file__).resolve().parent if not getattr(farm_gui.sys, 'frozen', False)
             else Path(farm_gui.sys.executable).resolve().parent)
    farm_gui.App = ArcherApp
    farm_gui.main()
