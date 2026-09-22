"""独立入口和配置，保留现有挂机方案的设置。"""
import os
import json
import tempfile
from pathlib import Path
import farm_gui
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from autofarm.version import ARCHER_VERSION
from autofarm import vision as V
from autofarm.custom_template import clipboard_image, save_template, check_owner
from name_template_gui import NameCropDialog

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
        for row, (label, key) in enumerate([('窗口标题', 'window_title'), ('独立模板', 'archer_profile'),
                                          ('角色名（可选）', 'archer_player_name')]):
            ttk.Label(box, text=label).grid(row=row, column=0, padx=8, pady=5)
            ttk.Entry(box, textvariable=self.vars[key], width=52).grid(row=row, column=1, padx=8, pady=5)
        ttk.Label(box, text='名字留空：使用原图片；填写名字：本地文字识别，自动生成本次备用图\n'
                  '右侧有猴子 → 长按 Shift；无猴子 → 松开\n'
                  '被击退 → 向平台内侧回位；钱币遮住身体 → 辅助识别头顶光圈',
                  justify='left').grid(row=3, column=0, columnspan=2, padx=8, pady=8, sticky='w')
        custom = ttk.LabelFrame(self, text='自己的名字图片（启用后优先使用）')
        custom.pack(fill='x', padx=12, pady=(0, 8))
        controls = ttk.Frame(custom)
        controls.pack(fill='x', padx=8, pady=5)
        ttk.Button(controls, text='粘贴图片', command=self._paste_name_template).pack(side='left')
        ttk.Button(controls, text='清除模板', command=self._clear_name_template).pack(side='left', padx=6)
        ttk.Label(controls, text='复制截图 → 粘贴 → 框选名字一行').pack(side='left')
        self.template_canvas = farm_gui.tk.Canvas(custom, width=560, height=58, background='#20242c',
                                                 highlightthickness=1, takefocus=True)
        self.template_canvas.pack(fill='x', padx=8, pady=3)
        self.template_canvas.bind('<Button-1>', lambda event: self.template_canvas.focus_set())
        self.template_status = farm_gui.tk.StringVar(value='未启用自定义模板')
        ttk.Label(custom, textvariable=self.template_status, wraplength=560).pack(anchor='w', padx=8, pady=5)
        self._template_loading = False
        self.vars['archer_player_name'].trace_add('write', lambda *args: self._refresh_name_template())
        self.bind('<Control-v>', self._paste_shortcut, add='+')
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
        self._template_loading = True
        if Path(farm_gui.CONFIG_FILE).exists():
            super()._load()
        else:
            self.on_archer_preset()
        self.vars['plan'].set('rope_archer')
        self.vision_on.set(True)
        self._template_loading = False
        self._refresh_name_template()
        self.title(f'冒险岛 · 绳边射手 v{ARCHER_VERSION}（向右攻击 / 先清怪后回位）')
        self._log(f'[版本] 绳边射手 v{ARCHER_VERSION}：确认右转 / 安全区先清怪后回位 / 粘贴模板')
        entry = farm_gui.sys.executable if getattr(farm_gui.sys, 'frozen', False) else __file__
        self._log(f'[程序] {Path(entry).resolve()}')

    def _paste_shortcut(self, event):
        # 文本输入框仍保留正常的文字粘贴；图片区域或按钮获得焦点时粘贴图片。
        if event.widget.winfo_class() in ('Entry', 'TEntry', 'Text', 'TCombobox', 'Spinbox', 'TSpinbox'):
            return
        self._paste_name_template()
        return 'break'

    def _paste_name_template(self):
        if self.alive:
            messagebox.showinfo('请先停止', '请先点「停止」，再更换名字模板。', parent=self)
            return
        try:
            image = clipboard_image()
            owner = self.vars['archer_player_name'].get().strip()
            def save(image):
                if self.alive:
                    raise ValueError('请先停止挂机再保存模板')
                path = save_template(image, owner)
                self._set_name_template(path, owner)
                self._log(f'[名字模板] 已保存并启用：{path}；请先点「试一下识别」')
            NameCropDialog(self, image, save)
        except (ValueError, OSError, Image.DecompressionBombError) as error:
            messagebox.showerror('无法粘贴图片', str(error), parent=self)

    def _set_name_template(self, path, owner):
        keys = ('archer_name_template', 'archer_template_owner')
        previous = [self.vars[key].get() for key in keys]
        self.vars[keys[0]].set(path)
        self.vars[keys[1]].set(owner)
        # 只有配置落盘成功才向玩家报告已保存；失败则恢复原配置。
        config = Path(farm_gui.CONFIG_FILE).resolve()
        temporary = None
        try:
            fd, temporary = tempfile.mkstemp(prefix='archer_', suffix='.tmp', dir=config.parent)
            with os.fdopen(fd, 'w', encoding='utf-8') as stream:
                json.dump(self._snapshot(), stream, ensure_ascii=False, indent=1)
            os.replace(temporary, config)
        except OSError:
            for key, value in zip(keys, previous):
                self.vars[key].set(value)
            raise
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)
            self._refresh_name_template()

    def _clear_name_template(self):
        if self.alive:
            messagebox.showinfo('请先停止', '请先点「停止」，再清除名字模板。', parent=self)
            return
        try:
            self._set_name_template('', '')
            self._log('[名字模板] 已清除选择，恢复输入名字 / 原图片模式')
        except OSError as error:
            messagebox.showerror('配置未保存', str(error), parent=self)

    def _refresh_name_template(self):
        if self._template_loading:
            return
        canvas = self.template_canvas
        canvas.delete('all')
        self._template_photo = None
        path = self.vars['archer_name_template'].get()
        if not path:
            canvas.create_text(280, 29, fill='#c4cbd5', text='点击这里后 Ctrl+V，或点击「粘贴图片」')
            self.template_status.set('未启用自定义模板；留空名字用原图，填写名字用 OCR')
            return
        try:
            check_owner(self.vars['archer_player_name'].get(), self.vars['archer_template_owner'].get())
            with Image.open(V.asset_path(path)) as im:
                scale = min(3, 550 / im.width, 52 / im.height)
                preview = im.resize((max(1, round(im.width*scale)), max(1, round(im.height*scale))), Image.Resampling.NEAREST)
                self._template_photo = ImageTk.PhotoImage(preview, master=self)
                canvas.create_image(280, 29, image=self._template_photo)
                self.template_status.set(f'已启用 {im.width} × {im.height} 原像素图片；清除后恢复文字 / 原图识别')
        except (ValueError, OSError, Image.DecompressionBombError) as error:
            self.template_status.set(f'模板暂不可用：{error}')


if __name__ == '__main__':
    os.chdir(Path(__file__).resolve().parent if not getattr(farm_gui.sys, 'frozen', False)
             else Path(farm_gui.sys.executable).resolve().parent)
    farm_gui.App = ArcherApp
    farm_gui.main()
