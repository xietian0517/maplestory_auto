"""玩家粘贴的名字图片：校验、原像素保存、唯一位置匹配。"""
import hashlib
import io
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageGrab

from . import vision as V
from .name_ocr import normalize_name


def clipboard_image():
    value = ImageGrab.grabclipboard()
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError('请只复制一张图片，或用截图工具复制名字区域')
        with Image.open(value[0]) as opened:
            return prepare_image(opened)
    if not isinstance(value, Image.Image):
        raise ValueError('剪贴板里没有图片，请先复制名字截图')
    return prepare_image(value)


def prepare_image(image):
    w, h = image.size
    if w < 1 or h < 1 or w > 8192 or h > 8192 or w * h > 16_000_000:
        raise ValueError('图片过大，请先截取名字附近的区域')
    # 透明像素不能成为不可见的匹配内容；使用和预览一致的深色底。
    if image.mode in ('RGBA', 'LA') or 'transparency' in image.info:
        rgba = image.convert('RGBA')
        base = Image.new('RGBA', rgba.size, (30, 30, 30, 255))
        return Image.alpha_composite(base, rgba).convert('RGB')
    return image.convert('RGB').copy()


def validate_template(image):
    w, h = image.size
    if not (12 <= w <= 256 and 6 <= h <= 40):
        raise ValueError('请紧贴完整名字框选一行文字（宽 12–256、高 6–40 像素），不要包含人物或称号')
    if np.asarray(image.convert('L')).std() < 8:
        raise ValueError('选区没有足够清晰的文字，请重新框选名字')


def save_template(image, owner='', root=None):
    image = prepare_image(image)
    validate_template(image)
    data = io.BytesIO()
    image.save(data, format='PNG')
    payload = data.getvalue()
    digest = hashlib.sha256(normalize_name(owner).encode('utf-8') + b'\0' + payload).hexdigest()[:24]
    relative = Path('user_templates/names') / f'{digest}.png'
    dest = (Path(root) if root is not None else V.program_dir()) / relative
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='name_', suffix='.tmp', dir=dest.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(payload)
        os.replace(temporary, dest)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return relative.as_posix()


def check_owner(player_name, owner):
    if normalize_name(player_name) != normalize_name(owner):
        raise ValueError('角色名已改变，请为新名字重新粘贴图片，或清除自定义模板后使用文字识别')


class CustomNameFinder(V.NameFinder):
    def __init__(self, path):
        super().__init__(path, .94)
        validate_template(Image.fromarray(cv2.cvtColor(self.template, cv2.COLOR_BGR2RGB)))
        self.ambiguous = False

    def find(self, frame):
        self.ambiguous = False
        h, w = self.template.shape[:2]
        if frame.shape[0] < h or frame.shape[1] < w:
            return None
        scores = cv2.matchTemplate(frame, self.template, cv2.TM_CCOEFF_NORMED)
        _, score, _, (x, y) = cv2.minMaxLoc(scores)
        if not np.isfinite(score) or score < self.threshold:
            return None
        scores[max(0, y-4):y+5, max(0, x-4):x+5] = -1
        if cv2.minMaxLoc(scores)[1] >= self.threshold:
            self.ambiguous = True
            return None
        return V.Hit(x + w / 2, y + h / 2, float(score))
