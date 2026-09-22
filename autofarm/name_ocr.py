"""输入角色名自动定位，再用当前画面的名字图片快速跟踪。仅使用本地模型。"""
from dataclasses import dataclass
import time
import unicodedata

import cv2
import numpy as np

from . import vision as V


def normalize_name(value):
    # 中文 OCR 可能插入空格；英文大小写不影响名字比较，不做近似字纠错。
    return ''.join(unicodedata.normalize('NFKC', value).split()).casefold()


def validate_name(value):
    value = value.strip()
    if any(unicodedata.category(c).startswith('C') for c in value) or len(value) > 24:
        raise ValueError('角色名请填写最多 24 个可见字符，不要包含换行或控制字符')
    return value


@dataclass(frozen=True)
class TextBox:
    text: str
    score: float
    left: float
    top: float
    right: float
    bottom: float


class LocalNameOCR:
    def __init__(self):
        from rapidocr import RapidOCR
        folder = V.asset_path('models/name_ocr')
        files = {'Det': 'PP-OCRv6_det_small.onnx', 'Rec': 'PP-OCRv6_rec_small.onnx',
                 'Cls': 'ch_ppocr_mobile_v2.0_cls_mobile.onnx'}
        for filename in files.values():
            if not (folder / filename).is_file():
                raise FileNotFoundError(f'缺少本地 OCR 模型：{folder / filename}；请保留发行包中的 models 文件夹')
        params = {'Global.log_level': 'error', 'Global.use_cls': False,
                  'Det.limit_side_len': 512, 'Det.limit_type': 'max',
                  'EngineConfig.onnxruntime.intra_op_num_threads': 2,
                  'EngineConfig.onnxruntime.inter_op_num_threads': 1}
        params.update({f'{kind}.model_path': str(folder / filename) for kind, filename in files.items()})
        self.engine = RapidOCR(params=params)

    def read(self, frame):
        scaled = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        result = self.engine(scaled, use_cls=False)
        if result.boxes is None or result.txts is None or result.scores is None:
            return []
        boxes = []
        for points, text, score in zip(result.boxes, result.txts, result.scores):
            p = np.asarray(points, dtype=float) / 2
            boxes.append(TextBox(text, float(score), float(p[:, 0].min()), float(p[:, 1].min()),
                                 float(p[:, 0].max()), float(p[:, 1].max())))
        return boxes


class TypedNameFinder:
    """缓存图片与输入名字绑定；不会用旧坐标代替当前帧检测。"""
    def __init__(self, name, reader=None):
        self.name = validate_name(name)
        if not self.name:
            raise ValueError('请输入角色名')
        self.target = normalize_name(self.name)
        self.reader = reader if reader is not None else LocalNameOCR()
        self.template = None
        self.offset = (0, 0)
        self.last_ocr = -float('inf')
        self.source = ''
        self.ambiguous = False

    def _picture(self, frame):
        if self.template is None:
            return None
        h, w = self.template.shape[:2]
        if h > frame.shape[0] or w > frame.shape[1]:
            return None
        scores = cv2.matchTemplate(frame, self.template, cv2.TM_CCOEFF_NORMED)
        _, score, _, (x, y) = cv2.minMaxLoc(scores)
        if not np.isfinite(score) or score < .94:
            return None
        # 同一张名字图若在两个位置同时出现，不能随便选一个。
        scores[max(0, y-4):y+5, max(0, x-4):x+5] = -1
        if cv2.minMaxLoc(scores)[1] >= .94:
            self.ambiguous = True
            return None
        self.source = '名字图片（OCR确认）'
        return V.Hit(x + self.offset[0], y + self.offset[1], score)

    def find(self, frame, now=None, allow_ocr=True):
        self.source, self.ambiguous = '', False
        hit = self._picture(frame)
        if hit is not None or self.ambiguous:
            return hit
        if not allow_ocr:
            return None
        now = time.monotonic() if now is None else now
        if now - self.last_ocr < .5:
            return None
        self.last_ocr = now
        boxes = [b for b in self.reader.read(frame)
                 if normalize_name(b.text) == self.target and b.score >= .95
                 and all(np.isfinite(v) for v in (b.score, b.left, b.top, b.right, b.bottom))
                 and 0 <= b.left < b.right <= frame.shape[1]
                 and 0 <= b.top < b.bottom <= frame.shape[0]
                 and 5 <= b.bottom - b.top <= 30]
        if len(boxes) != 1:
            self.ambiguous = len(boxes) > 1
            return None
        b = boxes[0]
        x, y = (b.left + b.right) / 2, (b.top + b.bottom) / 2
        left, top = max(0, int(b.left)), max(0, int(b.top))
        right, bottom = min(frame.shape[1], int(np.ceil(b.right))), min(frame.shape[0], int(np.ceil(b.bottom)))
        template = frame[top:bottom, left:right].copy()
        if template.std() >= 5:
            self.template = template
            self.offset = (x - left, y - top)
        self.source = '名字文字 OCR'
        return V.Hit(x, y, b.score)
