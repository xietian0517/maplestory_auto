"""Screenshot calibration and reusable template assets; no keyboard automation."""
import json
import os
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
import uuid

import numpy as np
from PIL import Image, ImageTk

from autofarm import vision as V
from autofarm.custom_template import prepare_image, validate_template
from autofarm.platform_guard import validate_profile, profile_asset
from name_template_gui import NameCropDialog, crop_box


def write_json(path, data):
    path = Path(path)
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temp, path)


def validate_monster(image):
    if not (8 <= image.width <= 256 and 6 <= image.height <= 256) or np.asarray(image.convert('L')).std() < 8:
        raise ValueError('请框选清晰的怪物特征（宽 8~256、高 6~256 像素）')


def save_calibration(image, boxes, monsters, name, player_name='', root=None):
    image = prepare_image(image)
    for key in ('anchor', 'player', 'platform', 'home', 'observe'):
        if key not in boxes:
            raise ValueError('请完成参照物、人物名字、平台、停靠和观察区域的标定')
    for box in [*boxes.values(), *monsters]:
        x1, y1, x2, y2 = box
        if not 0 <= x1 < x2 <= image.width or not 0 <= y1 < y2 <= image.height:
            raise ValueError('框选区域超出图片或为空')
    if not monsters:
        raise ValueError('请至少框选一张怪物模板')
    validate_template(image.crop(boxes['player']))
    anchor = image.crop(boxes['anchor'])
    if anchor.width < 8 or anchor.height < 8 or np.asarray(anchor.convert('L')).std() < 8:
        raise ValueError('请框选一个清晰且固定不动的平台图案作为参照物')
    for box in monsters:
        validate_monster(image.crop(box))
    ax = (boxes['anchor'][0]+boxes['anchor'][2])/2
    ay = (boxes['anchor'][1]+boxes['anchor'][3])/2
    def relative(box):
        return [v - (ax if i % 2 == 0 else ay) for i, v in enumerate(box)]
    platform, home = boxes['platform'], boxes['home']
    d = dict(name=name.strip() or '自定义守台', schema_version=1, anchor_template='anchor.png',
             player_template='player.png', player_name=player_name.strip(),
             anchor_threshold=.94, monster_threshold=.86,
             monster_templates=[f'monster_{i}.png' for i in range(len(monsters))],
             safe_left=platform[0]-ax, safe_right=platform[2]-ax,
             target_x=(home[0]+home[2])/2-ax, target_tolerance=(home[2]-home[0])/2,
             max_speed=350, player_roi=relative((max(0, platform[0]-45), 0, min(image.width, platform[2]+45), image.height)),
             monster_roi=relative(boxes['observe']), calibration_boxes=boxes,
             calibration_monsters=monsters, reference_image='reference.png')
    validate_profile(d)
    folder = (Path(root) if root is not None else V.program_dir()) / 'user_templates' / 'guards' / uuid.uuid4().hex[:12]
    folder.mkdir(parents=True)
    image.save(folder / 'reference.png')
    anchor.save(folder / 'anchor.png')
    image.crop(boxes['player']).save(folder / 'player.png')
    for filename, box in zip(d['monster_templates'], monsters):
        image.crop(box).save(folder / filename)
    write_json(folder / 'profile.json', d)
    return folder / 'profile.json'


def append_monster(profile, image):
    image = prepare_image(image)
    validate_monster(image)
    path = V.asset_path(profile)
    d = validate_profile(json.loads(path.read_text(encoding='utf-8')))
    filename = f'monster_{uuid.uuid4().hex[:12]}.png'
    d['monster_templates'] = [*d['monster_templates'], filename]
    validate_profile(d)
    image.save(profile_asset(path, filename))
    write_json(path, d)


def show_preview(parent, image, caption):
    window = tk.Toplevel(parent)
    window.title('自定义守台 · 识别预览（未发送按键）')
    window.transient(parent)
    image = prepare_image(image)
    scale = min(1, (window.winfo_screenwidth()-100)/image.width, (window.winfo_screenheight()-180)/image.height)
    photo = ImageTk.PhotoImage(image.resize((round(image.width*scale), round(image.height*scale))), master=window)
    label = ttk.Label(window, image=photo)
    label.image = photo
    label.pack(padx=8, pady=8)
    ttk.Label(window, text=caption, wraplength=800, padding=8).pack(anchor='w')
    ttk.Button(window, text='关闭', command=window.destroy).pack(pady=8)
    return window


class MonsterCropDialog(NameCropDialog):
    def __init__(self, parent, image, on_save):
        super().__init__(parent, image, on_save)
        self.title('追加怪物模板 · 框选身体、头部或光圈')
        self.winfo_children()[0].configure(text='框选一个怪物特征，避开血条、掉落物和技能特效；保存原始像素。')
        self.info.set('请框选怪物特征')
        if self.image.width <= 256 and self.image.height <= 256:
            self.start = (0, 0)
            self._select(self.display_size)

    def save(self):
        try:
            if self.selection is None:
                raise ValueError('请先框选怪物特征')
            selected = self.image.crop(self.selection)
            validate_monster(selected)
            self.on_save(selected)
        except (ValueError, OSError) as error:
            messagebox.showerror('未保存', str(error), parent=self)
            return
        self.destroy()


class GuardCalibrationDialog(tk.Toplevel):
    ROLES = [('anchor', '1. 固定参照物'), ('player', '2. 人物名字一行'),
             ('platform', '3. 平台左右边界'), ('home', '4. 期望停靠范围'), ('observe', '5. 怪物观察区域')]
    TIPS = {'anchor': '框选附近固定不动的平台纹理，避免树叶、人物或特效。',
            'player': '紧贴自己的完整名字框一行，左右留白对称。',
            'platform': '框出角色可站立的平台；左右边框就是移动边界，勿包含断崖。',
            'home': '框出期望停留的一段范围，宽至少 40px，左右离平台边界各超过 20px。',
            'observe': '框出左右需要监视的怪物区域，排除其他台子。',
            'monster': '拖动框选一个怪物身体、头部或光圈；可以连续添加多个框。'}

    def __init__(self, parent, image, on_save, player_name='', existing=None):
        super().__init__(parent)
        self.title('自定义守台 · 标定平台与怪物')
        self.transient(parent)
        self.image = prepare_image(image)
        self.on_save, self.player_name = on_save, player_name
        self.boxes = dict((existing or {}).get('calibration_boxes', {}))
        self.monsters = list((existing or {}).get('calibration_monsters', []))
        self.extra_paths = []
        if existing and existing.get('_profile_path'):
            self.extra_paths = [profile_asset(Path(existing['_profile_path']), filename)
                                for filename in existing['monster_templates'][len(self.monsters):]]
        self.active, self.start, self.selection = 'anchor', None, None
        self.name = tk.StringVar(value=(existing or {}).get('name', '我的守台方案'))
        top = ttk.Frame(self)
        top.pack(fill='x', padx=10, pady=8)
        ttk.Label(top, text='方案名').pack(side='left')
        ttk.Entry(top, textvariable=self.name, width=32).pack(side='left', padx=8)
        buttons = ttk.Frame(self)
        buttons.pack(fill='x', padx=8)
        for role, label in self.ROLES:
            ttk.Button(buttons, text=label, command=lambda r=role: self.choose(r)).pack(side='left', padx=2)
        self.tip = tk.StringVar(value=self.TIPS[self.active])
        ttk.Label(self, textvariable=self.tip, padding=8).pack(anchor='w')
        scale = min(1, (self.winfo_screenwidth()-100)/self.image.width, (self.winfo_screenheight()-290)/self.image.height)
        self.display_size = (round(self.image.width*scale), round(self.image.height*scale))
        self.photo = ImageTk.PhotoImage(self.image.resize(self.display_size), master=self)
        self.canvas = tk.Canvas(self, width=self.display_size[0], height=self.display_size[1], cursor='crosshair')
        self.canvas.pack(padx=8)
        self.canvas.create_image(0, 0, anchor='nw', image=self.photo)
        self.canvas.bind('<ButtonPress-1>', self.press)
        self.canvas.bind('<B1-Motion>', self.drag)
        self.canvas.bind('<ButtonRelease-1>', self.release)
        self.summary = tk.StringVar()
        ttk.Label(self, textvariable=self.summary, padding=8).pack(anchor='w')
        bottom = ttk.Frame(self)
        bottom.pack(fill='x', padx=8, pady=8)
        ttk.Button(bottom, text='6. 添加怪物框', command=lambda: self.choose('monster')).pack(side='left')
        ttk.Button(bottom, text='撤销最后一个怪物', command=self.undo).pack(side='left', padx=6)
        ttk.Button(bottom, text='保存为新方案', command=self.save).pack(side='right')
        ttk.Button(bottom, text='取消', command=self.destroy).pack(side='right', padx=6)
        self.draw()
        self.grab_set()

    def choose(self, role):
        self.active = role
        self.tip.set(self.TIPS[role])

    def press(self, event):
        self.start = (event.x, event.y)

    def drag(self, event):
        if self.start:
            self.selection = crop_box(self.start, (event.x, event.y), self.display_size, self.image.size)
            self.draw()

    def release(self, event):
        self.drag(event)
        if self.selection:
            if self.active == 'monster':
                self.monsters.append(self.selection)
            else:
                self.boxes[self.active] = self.selection
            self.selection, self.start = None, None
            self.draw()

    def undo(self):
        if self.monsters:
            self.monsters.pop()
        self.draw()

    def draw(self):
        self.canvas.delete('boxes')
        sx, sy = self.display_size[0]/self.image.width, self.display_size[1]/self.image.height
        items = [(k, box) for k, box in self.boxes.items()] + [('monster', b) for b in self.monsters]
        if self.selection:
            items.append(('selection', self.selection))
        for name, (x1, y1, x2, y2) in items:
            color = {'platform': '#ff5050', 'home': '#40ffb0', 'monster': '#ffff40'}.get(name, '#40c8ff')
            self.canvas.create_rectangle(x1*sx, y1*sy, x2*sx, y2*sy, outline=color, width=2, tags='boxes')
            self.canvas.create_text(x1*sx+3, y1*sy+3, text=name, fill=color, anchor='nw', tags='boxes')
        self.summary.set(f'已标定 {len(self.boxes)}/5 项；怪物框 {len(self.monsters)} 个，保留追加图片 {len(self.extra_paths)} 张。')

    def save(self):
        try:
            path = save_calibration(self.image, self.boxes, self.monsters, self.name.get(), self.player_name)
            for extra in self.extra_paths:
                with Image.open(extra) as image:
                    append_monster(str(path), image)
            self.on_save(path)
        except (ValueError, OSError) as error:
            messagebox.showerror('方案未保存', str(error), parent=self)
            return
        self.destroy()
