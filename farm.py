# -*- coding: utf-8 -*-
"""冒险岛挂机入口。所有可调参数都在下面 CONFIG 里；改完保存，重新运行即可。

运行：  .\\.venv\\Scripts\\python.exe farm.py
开始后先暂停：切到游戏窗口，按 F12 开始/暂停，F11 退出。
注意：本客户端只认 SendInput（前台注入），运行时游戏窗口必须在最前面；
     切出游戏会自动挂起、不发任何键，切回来自动继续。
     游戏由 GameGuard 拉起、是管理员权限，所以打开终端时也要「以管理员身份运行」，
     否则按键会被 Windows 静默丢弃，程序看着在跑但游戏毫无反应。

默认方案 vision_jump：每隔几秒截一张图，用名字牌（如 CatApril）找到主角在画面里的 x，
判断她在平台靠左还是靠右，然后一直朝那个反方向打；跨过中线才换方向，不再左右各一下。
这样每步都往远离她那一端的方向走，贴边时先往中间挪，避免走出平台掉下去。
名字牌模板和平台范围用 farm_gui.py 的「截屏标定」生成，没标定会自动退回纯计时打法。
"""
from dataclasses import dataclass

from autofarm.bot import Bot, BotStopped
from autofarm import plans


@dataclass(frozen=True)
class Config:
    # ---------- 窗口与方案 ----------
    window_title: str = '冒险岛怀旧服'      # 游戏窗口标题里唯一的一段字，支持中文
    plan: str = 'vision_jump'            # vision_jump（截图判断版，推荐）/ random_jump / fixed_jump / static_cast

    # ---------- 按键 ----------
    attack_key: str = 'shift'             # 攻击/技能键（魔法双击=shift，也可填 a-z/0-9/ctrl/space 等）
    jump_key: str = 'alt'                 # 跳跃键（怀旧服默认 Alt=跳）
    potion_key: str | None = None         # 蓝药键，不喝药；想开就填如 'end'
    buff_key: str | None = 'home'         # Buff 键，默认 Home；不想定时加 Buff 就在界面关掉或留空

    # ---------- 节奏（单位：秒，二元组表示随机区间）----------
    attacks_per_side: int = 1             # 每个方向跳攻几下（1=右一下左一下）
    jump_hold_secs: tuple = (0.050, 0.090)    # 跳跃键按住时长（至少 50ms，确保 Alt 被注册成跳跃）
    jump_rise_secs: tuple = (0.050, 0.080)    # 起跳后等多久再出招（没跳起来就调大，空中出招太慢就调小）
    key_hold_secs: tuple = (0.030, 0.080)     # 攻击键按住时长
    attack_gap_secs: tuple = (0.250, 0.450)   # 两次跳攻之间的间隔（放慢版，含落地）
    move_secs: tuple = (0.040, 0.120)         # 单向移动时长，每轮随机，一轮内左右同值（回原位）
    settle_secs: tuple = (0.030, 0.060)        # 移动后停稳等待
    switch_gap_secs: tuple = (0.350, 0.550)    # 左右换边间隔（防键冲突、更自然）

    # ---------- 喝药（potion_key 不为 None 时生效）----------
    potion_every: int = 15                # 每多少下攻击喝1瓶（旧版临界：15下≈300蓝≈1瓶药）
    potion_pause_secs: tuple = (0.6, 1.2)        # 喝药后停顿

    # ---------- 定时 Buff（buff_key 不为 None 时生效，按时间不按攻击次数）----------
    buff_every_secs: tuple = (60.0, 90.0)      # 每隔多久按一次 Buff（随机区间，单位秒）
    buff_pause_secs: tuple = (0.3, 0.6)        # 按完 Buff 键后的停顿，等技能动作播完

    # ---------- random_jump 方案专用（更随机的版本）----------
    extra_attack_prob: float = 0.15       # 某一边多打1下的概率（0~1，0.15=约7次里1次）
    hop_prob: float = 0.05                # 插入一次纯跳不攻击的概率
    idle_prob: float = 0.08               # 整轮打完后随机发呆的概率
    idle_secs: tuple = (0.5, 1.5)         # 发呆时长区间

    # ---------- static_cast 方案专用 ----------
    micro_move_every: int = 25             # 站桩打法每多少下做一次左右微调（25~40 比较稳）

    # ---------- vision_jump 专用：截图找名字牌，判断主角靠左还是靠右 ----------
    vision_enabled: bool = True           # 关掉就退回纯计时打法，不截图
    name_template: str = 'assets/name_CatApril.png'   # 名字牌模板；界面「截屏标定」可重新生成
    match_threshold: float = 0.75         # 匹配阈值：真名牌接近 1.0，聊天栏文字约 0.6
    vision_interval: float = 4.0          # 隔几秒截一张图
    platform_left: int | None = None      # 平台左端（客户区像素），标定得到，可留空
    platform_right: int | None = None     # 平台右端，留空就按整个画面判断左右
    edge_margin: int = 60                 # 离两端不足这么多像素算贴边，先往中间挪
    rescue_scale: float = 2.0             # 贴边那一轮移动时长放大倍数（净往中间挪）
    archer_profile: str = 'templates/rope_archer/profile.json'
    archer_player_name: str = ''           # 射手：留空匹配图片，填写角色名启用本地 OCR


def main():
    cfg = Config()
    print('=' * 52)
    print(' 冒险岛挂机（SendInput 前台版）')
    print(' F12 开始/暂停    F11 退出    失焦自动挂起')
    print('=' * 52)
    try:
        bot = Bot(cfg.window_title)
    except Exception as e:
        print(f'[错误] {e}')
        return
    if cfg.plan not in plans.PLANS:
        print(f'[错误] 未知方案 {cfg.plan}，可选：{", ".join(plans.PLANS)}')
        return
    print(f'绑定窗口：{bot.window_name}')
    print(f'当前方案：{cfg.plan}（每边{cfg.attacks_per_side}下，'
          f'攻击间隔{cfg.attack_gap_secs[0] * 1000:.0f}~{cfg.attack_gap_secs[1] * 1000:.0f}ms）')
    if cfg.buff_key:
        print(f'定时 Buff：每 {cfg.buff_every_secs[0]:.0f}~{cfg.buff_every_secs[1]:.0f} 秒按一次 {cfg.buff_key}')
    print('已暂停：切到游戏窗口后按 F12 开始。')
    try:
        plans.PLANS[cfg.plan](bot, cfg)
    except BotStopped:
        print('[退出] 收到 F11。')
    except KeyboardInterrupt:
        print('[退出] Ctrl+C。')
    finally:
        bot.release_all()
        print('已兜底抬起所有按键，程序结束。')


if __name__ == '__main__':
    main()
