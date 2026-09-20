# -*- coding: utf-8 -*-
"""改进版自动标定：专门找角色头上白底黑字的名字牌（CatApril 式）。

策略：
1) 先在画面中心 60% 区域找（名字牌一般在角色头上，不会太靠边）
2) 找白底（亮）上的深色文字（黑/深灰）——和角色名牌的实际样式匹配
3) 挑最符合 "白底黑字 + 宽高比 3~7 + 白色占比 40~80%" 的那块
"""
from pathlib import Path

import cv2
import numpy as np

from autofarm.vision import program_dir


def find_nameplate(frame_bgr):
    h, w = frame_bgr.shape[:2]
    # 先裁中心区域（跳过左上角小地图、顶部商城/任务栏、右侧聊天栏）
    margin_x = int(w * 0.08)
    margin_top = int(h * 0.12)
    margin_bot = int(h * 0.20)
    roi = frame_bgr[margin_top:h - margin_bot, margin_x:w - margin_x]

    # 白底：R/G/B 都很高且接近（饱和度低）
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    white_bg = cv2.inRange(hsv, (0, 0, 170), (180, 50, 255))   # 白/近白
    dark_fg = cv2.inRange(hsv, (0, 0, 0), (180, 255, 90))     # 黑/深灰文字

    # 白色背景的连通区域
    contours, _ = cv2.findContours(white_bg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        area = bw * bh
        if area < 200 or area > 30000:
            continue
        aspect = bw / max(bh, 1)
        if aspect < 2.5 or aspect > 10:
            continue
        # 在这个白底框里，深色像素占比（文字密度）
        region_dark = dark_fg[y:y + bh, x:x + bw]
        dark_ratio = region_dark.mean() / 255.0
        if dark_ratio < 0.15 or dark_ratio > 0.60:
            continue
        candidates.append((bw * bh, aspect, dark_ratio, x, y, bw, bh))

    # 打分：面积适中 + 宽高比接近 5 + 深色占比 ~0.35
    def score(c):
        area, aspect, dark_ratio = c[0], c[1], c[2]
        area_score = max(0, 1 - abs(area - 3000) / 5000)   # ~3000px 理想
        aspect_score = max(0, 1 - abs(aspect - 5) / 5)
        dark_score = max(0, 1 - abs(dark_ratio - 0.30) / 0.30)
        return area_score * 0.4 + aspect_score * 0.3 + dark_score * 0.3

    candidates.sort(key=score, reverse=True)
    print(f'候选数: {len(candidates)}  Top 3:')
    for i, c in enumerate(candidates[:3]):
        s = score(c)
        print(f'  #{i + 1} 得分 {s:.3f}  位置 ({c[3] + margin_x},{c[4] + margin_top})  '
              f'{c[5]}x{c[6]}  面积{c[0]} 宽高比{c[1]:.2f} 文字密度{c[2]:.2f}')
        yield s, (c[3] + margin_x, c[4] + margin_top, c[5], c[6])


def main():
    assets = program_dir() / 'assets'
    shot = assets / 'game_screenshot.png'
    if not shot.exists():
        print(f'没有游戏截图：{shot}')
        return
    frame = cv2.imread(str(shot))
    print(f'截图尺寸 {frame.shape[1]}x{frame.shape[0]}')

    results = list(find_nameplate(frame))
    if not results:
        print('没找到名字牌候选')
        return
    best_score, (bx, by, bw, bh) = results[0]

    # 四周加 2px 余量
    pad = 2
    bx = max(0, bx - pad); by = max(0, by - pad)
    bw = min(frame.shape[1] - bx, bw + 2 * pad)
    bh = min(frame.shape[0] - by, bh + 2 * pad)
    best = frame[by:by + bh, bx:bx + bw]

    out = assets / 'name_CatApril.png'
    cv2.imencode('.png', best)[1].tofile(str(out))
    print(f'\n模板已保存：{out}  尺寸 {bw}x{bh}')

    # 验证
    from autofarm import vision as V
    finder = V.NameFinder(str(out), 0.75)
    hit = finder.find(frame)
    if hit:
        print(f'匹配验证：x={hit.x:.0f} y={hit.y:.0f}  得分={hit.score:.3f}  '
              f'{"OK" if hit.score > 0.85 else "偏低，可能要重标"}')
    else:
        print('匹配验证失败')


if __name__ == '__main__':
    main()
