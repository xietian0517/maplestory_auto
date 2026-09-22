# -*- coding: utf-8 -*-
"""独立加Buff工具：不开挂机方案，只负责定时按 Buff 键。

和挂机程序相互独立，可以单独跑（比如手动玩的时候挂着补Buff）：
    .\\.venv\\Scripts\\python.exe buff_timer.py

控制：F12 开始/暂停（启动后是暂停态），F11 退出。
保护：游戏不在前台自动挂起不发键，回到前台自动继续；
     游戏是管理员权限时本程序也必须管理员运行（绑定窗口时会检查）。

不共享CD：下面每个槽位各自独立计时，互不影响；
暂停期间计时照走，恢复后会先各补按一次（暂停久了Buff本来也该续了）。
要加更多Buff键，照 BUFF_SLOTS 的格式再添一行即可。
"""
import time

from autofarm.bot import Bot, BotStopped
from autofarm.blocks import Interval, rnd

WINDOW_TITLE = '冒险岛怀旧服'      # 游戏窗口标题里唯一的一段字
KEY_HOLD_SECS = (0.030, 0.080)     # Buff键按下停留时长（毫秒级，跟攻击键一致）
POLL_SECS = 0.1                    # 空闲轮询间隔（秒）

# 每行 = (键名, 间隔秒区间(min, max), 按完停顿秒区间)
# 键名：home/ins/insert/delete/end/pgup/a-z/0-9/f8 等，小写
BUFF_SLOTS = [
    ('home', (60.0, 90.0), (0.3, 0.6)),     # Buff 1：默认约 60~90 秒一次
    ('ins', (120.0, 180.0), (0.3, 0.6)),    # Buff 2：默认约 2~3 分钟一次
]


def main():
    print('=' * 52)
    print(' 独立加Buff工具（SendInput 前台版）')
    print(' F12 开始/暂停    F11 退出    失焦自动挂起')
    print('=' * 52)
    try:
        bot = Bot(WINDOW_TITLE)
    except Exception as e:
        print(f'[错误] {e}')
        return
    slots = [(key, Interval(rng), pause) for key, rng, pause in BUFF_SLOTS]
    print(f'绑定窗口：{bot.window_name}')
    for key, timer, _ in slots:
        print(f'  Buff 键 {key}：每 {timer.rng[0]:.0f}~{timer.rng[1]:.0f} 秒一次')
    print('已暂停：切到游戏窗口后按 F12 开始。')
    try:
        while True:
            for key, timer, pause in slots:
                if timer.due():
                    bot.log(f'[Buff] 按 {key}')
                    bot.tap(key, rnd(KEY_HOLD_SECS))
                    bot.wait(rnd(pause))      # 等技能动作播完再看下一个槽位
            bot.wait(POLL_SECS)               # 空闲轮询：F12/F11/失焦都在这响应
    except BotStopped:
        print('[退出] 收到 F11。')
    except KeyboardInterrupt:
        print('[退出] Ctrl+C。')
    finally:
        bot.release_all()
        print('已兜底抬起所有按键，程序结束。')


if __name__ == '__main__':
    main()
