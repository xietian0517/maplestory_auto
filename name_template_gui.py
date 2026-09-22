"""名字模板裁剪对话框。只处理粘贴图片，不截图、不控制游戏。"""
import tkinter as tk
from tkinter import ttk, messagebox

from PIL import Image, ImageTk

from autofarm.custom_template import prepare_image, validate_template


def crop_box(start, end, display_size, image_size):
    dw, dh = display_size
    iw, ih = image_size
    xs = [min(iw, max(0, round(x * iw / dw))) for x in (start[0], end[0])]
    ys = [min(ih, max(0, round(y * ih / dh))) for y in (start[1], end[1])]
    return min(xs), min(ys), max(xs), max(ys)


class NameCropDialog(tk.Toplevel):
    def __init__(self, parent, image, on_save):
        super().__init__(parent)
        self.title('粘贴名字模板 · 框选完整名字')
        self.transient(parent)
        self.resizable(False, False)
        self.image = prepare_image(image)
        self.on_save = on_save
        self.selection = None
        self.start = None
        ttk.Label(self, text='拖动框选名字一行，左右留白尽量对称；不要包含人物、称号或宠物名。\n'
                  '仅放大预览，保存时保留游戏截图原始像素。', padding=10).pack(anchor='w')
        max_width = min(840, self.winfo_screenwidth() - 100)
        max_height = min(440, self.winfo_screenheight() - 230)
        scale = min(4, max_width / self.image.width, max_height / self.image.height)
        self.display_size = (max(1, round(self.image.width * scale)), max(1, round(self.image.height * scale)))
        self.photo = ImageTk.PhotoImage(self.image.resize(self.display_size, Image.Resampling.NEAREST), master=self)
        self.canvas = tk.Canvas(self, width=self.display_size[0], height=self.display_size[1],
                                highlightthickness=0, cursor='crosshair')
        self.canvas.pack(padx=10)
        self.canvas.create_image(0, 0, anchor='nw', image=self.photo)
        self.canvas.bind('<ButtonPress-1>', self._press)
        self.canvas.bind('<B1-Motion>', self._drag)
        self.canvas.bind('<ButtonRelease-1>', self._drag)
        self.info = tk.StringVar(value='请框选名字区域')
        ttk.Label(self, textvariable=self.info, padding=8).pack(anchor='w')
        buttons = ttk.Frame(self)
        buttons.pack(fill='x', padx=10, pady=(0, 10))
        ttk.Button(buttons, text='保存并使用模板', command=self.save).pack(side='left')
        ttk.Button(buttons, text='取消', command=self.destroy).pack(side='right')
        self.bind('<Escape>', lambda event: self.destroy())
        self.bind('<Return>', lambda event: self.save())
        # 已经截成单行名字的小图可直接保存，大图必须重新框选。
        if 12 <= self.image.width <= 256 and 6 <= self.image.height <= 40:
            self.start = (0, 0)
            self._select(self.display_size)
        self.grab_set()
        self.canvas.focus_set()

    def _press(self, event):
        self.start = (event.x, event.y)
        self._select(self.start)

    def _drag(self, event):
        if self.start is not None:
            self._select((event.x, event.y))

    def _select(self, end):
        self.selection = crop_box(self.start, end, self.display_size, self.image.size)
        left, top, right, bottom = self.selection
        sx, sy = self.display_size[0] / self.image.width, self.display_size[1] / self.image.height
        self.canvas.delete('selection')
        self.canvas.create_rectangle(left*sx, top*sy, right*sx, bottom*sy,
                                     outline='#00e0a0', width=2, tags='selection')
        self.canvas.create_line((left+right)*sx/2, top*sy, (left+right)*sx/2, bottom*sy,
                                fill='#00e0a0', dash=(3, 3), tags='selection')
        self.info.set(f'原始选区：{right-left} × {bottom-top} 像素；虚线为名字中心')

    def save(self):
        try:
            if self.selection is None:
                raise ValueError('请先拖动框选完整名字')
            selected = self.image.crop(self.selection)
            validate_template(selected)
            self.on_save(selected)
        except (ValueError, OSError) as error:
            messagebox.showerror('模板未保存', str(error), parent=self)
            return
        self.destroy()
