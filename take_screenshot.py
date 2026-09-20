# -*- coding: utf-8 -*-
"""自动截一张当前游戏窗口的画面，方便你用画图框出 CatApril 名字牌。

用法：  .\\.venv\\Scripts\\python.exe take_screenshot.py
输出：  assets/game_screenshot.png（截完会告诉你完整路径）
后续：  用任何看图工具打开，框住 CatApril 那几行白字，裁出来保存为 assets/name_CatApril.png 即可。
"""
from pathlib import Path

import cv2

from autofarm import winapi
from autofarm.vision import capture, program_dir


def main():
    title = '冒险岛怀旧服'
    api = winapi.WinApi()
    hwnd, name = api.find_window(title)
    rect = api.client_rect(hwnd)
    print(f'绑定窗口：{name}，客户区 {rect}')
    frame = capture(rect)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    out_dir = program_dir() / 'assets'
    out_dir.mkdir(exist_ok=True)
    out = out_dir / 'game_screenshot.png'
    cv2.imwrite(str(out), frame_bgr)
    print(f'已保存：{out}  尺寸 {frame_bgr.shape[1]}x{frame_bgr.shape[0]}')


if __name__ == '__main__':
    main()
