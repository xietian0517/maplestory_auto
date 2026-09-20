# -*- coding: utf-8 -*-
"""快速标定名字牌模板：运行后会截一张游戏窗口，请用鼠标在弹出的窗口里框一下你的名字牌。

用法：  .\\.venv\\Scripts\\python.exe quick_calibrate.py        # 默认找标题含「冒险岛怀旧服」的窗口
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from autofarm import winapi
from autofarm.vision import capture, program_dir


def main():
    title = '冒险岛怀旧服'
    cfg_name = 'assets/name_CatApril.png'
    try:
        api = winapi.WinApi()
        hwnd, name = api.find_window(title)
        rect = api.client_rect(hwnd)
        print(f'绑定窗口：{name}，客户区 {rect}')
    except Exception as e:
        print(f'找不到游戏窗口：{e}')
        sys.exit(1)

    print('3 秒后截一张图… 请确保游戏窗口完整可见')
    time.sleep(3)
    frame = capture(rect)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    print('在弹出的窗口里，按住鼠标左键拖一个框框住你的名字牌（例如 CatApril），回车确认。')
    print('框完回车即继续；按 r 重选；按 q 放弃。')
    x, y, w, h = cv2.selectROI('框选你的名字牌（回车确认）', frame_bgr,
                               showCrosshair=False, fromCenter=False)
    cv2.destroyAllWindows()
    if w == 0 or h == 0:
        print('未选，放弃。')
        sys.exit(0)

    out_dir = program_dir() / 'assets'
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / 'name_CatApril.png'
    roi = frame_bgr[y:y + h, x:x + w]
    cv2.imencode('.png', roi)[1].tofile(str(out_path))       # 中文路径用 tofile
    print(f'名字牌模板已保存到: {out_path}  ({w}x{h})')

    frame2 = frame_bgr.copy()
    cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imwrite(str(out_dir / 'calibrated_full.png'), frame2)
    print(f'完整截图已保存: {out_dir / "calibrated_full.png"}（方便你核对位置）')

    print('\n现在要标定平台范围吗？(y/n，默认 n)')
    ans = input().strip().lower() or 'n'
    platform_left, platform_right = None, None
    if ans == 'y':
        print('在弹出的窗口里框住平台可站立范围的左右两端，回车确认。')
        xl, yl, wl, hl = cv2.selectROI('框选平台左端 -> 回车', frame_bgr,
                                        showCrosshair=False, fromCenter=False)
        cv2.destroyAllWindows()
        xr, yr, wr, hr = cv2.selectROI('框选平台右端 -> 回车', frame_bgr,
                                        showCrosshair=False, fromCenter=False)
        cv2.destroyAllWindows()
        if wl > 0 and wr > 0:
            platform_left = xl + wl // 2
            platform_right = xr + wr // 2
            print(f'平台左端 x={platform_left}, 右端 x={platform_right}')

    print(f'\n标定完成。现在跑 vision_jump 即可：')
    print(f'  控制台版:  .\\.venv\\Scripts\\python.exe farm.py')
    print(f'  界面版:    .\\.venv\\Scripts\\python.exe farm_gui.py')
    print(f'模板路径（已写死在 farm.py Config.name_template）: assets/name_CatApril.png')
    if platform_left is not None:
        print(f'平台范围：platform_left={platform_left}, platform_right={platform_right}')


if __name__ == '__main__':
    main()
