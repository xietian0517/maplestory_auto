"""截图判断：靠名字牌找到主角，决定先往哪边打。

实测结论（1200x700 左右的客户区）：名字牌是静态白字，比人物模型和动作帧稳得多，
一张截图做一次模板匹配就能拿到主角的 x；同一张图里聊天栏文字只有 0.6 分左右，
真名牌接近 1.0，阈值很好分。

只截屏 + 图像匹配：不读内存、不注入、不修改客户端。需要 opencv 和 mss。
"""
from dataclasses import dataclass
from pathlib import Path
import sys

import cv2
import mss
import numpy as np


@dataclass(frozen=True)
class Hit:
    """名字牌在客户区里的位置（像素）。"""

    x: float
    y: float
    score: float


def program_dir():
    """程序目录：打包后是 exe 所在目录，源码运行是项目根目录。"""
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def asset_path(value):
    """模板等素材的相对路径按程序目录解析，不跟着工作目录跑。"""
    path = Path(value)
    return path if path.is_absolute() else program_dir() / path


def capture(region):
    """截取窗口客户区（region 来自 WinApi.client_rect），返回 BGR 图像。"""
    with mss.mss() as screen:
        return np.array(screen.grab(region))[:, :, :3].copy()


def choose_direction(x, left, right, margin):
    """判断主角在平台偏左还是偏右，返回 (先打哪边, 是否贴边)。

    偏左就一直往右打、偏右就一直往左打：方向保持不变，直到她跨过中线或贴到边。
    离两端不足 margin 像素算贴边，方向一律指向平台中间，绝不再往外走。
    x 为 None 表示这次没识别到，返回 (None, False)，由调用方沿用旧方向。
    """
    if x is None:
        return None, False
    if left is None or right is None or right <= left:
        raise ValueError('平台左右边界无效，请重新标定')
    if x < left + margin:
        return 'right', True
    if x > right - margin:
        return 'left', True
    return ('right' if x < (left + right) / 2 else 'left'), False


def annotate(frame, hit, text=''):
    """画个圈标出识别位置，方便肉眼确认（文字请用 ASCII）。"""
    view = frame.copy()
    if hit is not None:
        cv2.circle(view, (int(hit.x), int(hit.y)), 28, (0, 255, 0), 2)
    if text:
        cv2.putText(view, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return view


class NameFinder:
    """用名字牌模板在截图里定位主角。"""

    def __init__(self, path, threshold):
        self.path = asset_path(path)
        self.name = self.path.name
        self.threshold = float(threshold)
        self.template = cv2.imdecode(np.fromfile(str(self.path), dtype=np.uint8),
                                     cv2.IMREAD_COLOR)
        if self.template is None:
            raise ValueError(f'名字模板读不出来: {self.path}')
        if self.template.std() < 5:
            raise ValueError(f'名字模板没有纹理（是不是框到空白了）: {self.path}')

    def find(self, frame):
        """返回得分最高且过阈值的 Hit；没找到返回 None。"""
        height, width = self.template.shape[:2]
        if height > frame.shape[0] or width > frame.shape[1]:
            return None
        scores = cv2.matchTemplate(frame, self.template, cv2.TM_CCOEFF_NORMED)
        _, score, _, (x, y) = cv2.minMaxLoc(scores)
        if not np.isfinite(score) or score < self.threshold:
            return None
        return Hit(x + width / 2, y + height / 2, float(score))
