"""挂机方案：用 blocks 里的积木拼出来的循环打法。

写新方案就是照着文字描述顺序排积木，例如：
    右移一次 -> 原地打3下 -> 停250~450ms -> 左移一次 -> 原地打3下 -> 循环
写完注册到下面的 PLANS 字典，再把 farm.py 里的 plan 名字换掉即可。
"""
import random
import time

from . import blocks as B
from . import vision as V

SIDES = ('right', 'left')
SIDE_CN = {'right': '右', 'left': '左'}


def random_sides():
    """每轮随机先后手：一半概率先右后左，一半先左后右。"""
    first = random.choice(SIDES)
    second = 'left' if first == 'right' else 'right'
    return first, second


def make_buff(cfg):
    """配置了 Buff 键就返回一个定时器，否则 None。"""
    return B.Interval(cfg.buff_every_secs) if cfg.buff_key else None


def maybe_buff(bot, cfg, buff):
    """每轮开头问一句：到点了就按一下 Buff 键。"""
    if buff is not None and buff.due():
        B.cast_buff(bot, cfg.buff_key, cfg.key_hold_secs, cfg.buff_pause_secs)


class BalancedDeck:
    """平衡牌堆：每 pair_count 轮（=2*pair_count 次攻击）洗一副牌。

    一副牌内先右/先左的轮次数最多相差1，因此任意一副牌打完，
    左右攻击次数严格各半；轮次顺序随机，不会老往一边起手。
    pair_count=5 即「10次攻击内5左5右」。
    """

    def __init__(self, pair_count=5):
        self.pair_count = pair_count
        self.deck = []

    def _shuffle(self):
        rights = self.pair_count // 2
        lefts = self.pair_count - rights
        # 奇数副时随机决定哪边多一次
        if self.pair_count % 2 == 1 and random.random() < 0.5:
            rights, lefts = lefts, rights
        self.deck = ['right'] * rights + ['left'] * lefts
        random.shuffle(self.deck)

    def next_pair(self):
        """返回本轮 (先打方向, 后打方向)。"""
        if not self.deck:
            self._shuffle()
        first = self.deck.pop()
        second = 'left' if first == 'right' else 'right'
        return first, second


def fixed_jump(bot, cfg):
    """【保存版】固定先右后左各一次：移动一步 -> 停稳 -> 原地跳攻 -> 换边间隔。

    左右移动共用同一个随机时长，打完两边正好回原位；移动本身就是防检测微调。
    """
    potion = B.Every(cfg.potion_every) if cfg.potion_key else None
    buff = make_buff(cfg)
    round_no = 0
    while True:
        round_no += 1
        maybe_buff(bot, cfg, buff)
        move_secs = B.rnd(cfg.move_secs)          # 本轮左右共用同一时长，保证回原位
        for side in SIDES:                        # 固定先右后左，各一次
            B.move(bot, side, move_secs)
            B.settle(bot, cfg.settle_secs)
            for i in range(1, cfg.attacks_per_side + 1):
                B.jump_attack(bot, cfg.jump_key, cfg.attack_key,
                              cfg.jump_hold_secs, cfg.jump_rise_secs,
                              cfg.key_hold_secs,
                              f'[第{round_no}轮]{SIDE_CN[side]}跳攻({i}/{cfg.attacks_per_side})')
                B.wait(bot, cfg.attack_gap_secs)
                if potion is not None and potion.hit():
                    B.drink_potion(bot, cfg.potion_key, cfg.key_hold_secs,
                                   cfg.potion_pause_secs)
            B.switch_side(bot, cfg.switch_gap_secs)   # 换方向间隔 250~450ms


def random_jump(bot, cfg):
    """【随机版】在 fixed_jump 基础上加大随机性：

    - 先后手随机（一半先右一半先左）
    - 每边 1 下为主，小概率多打 1 下（extra_attack_prob）
    - 小概率插入一次纯跳不攻击（hop_prob，模拟跳跃走位）
    - 小概率整轮打完发呆一会儿（idle_prob）
    每轮左右移动仍共用同一随机时长，保证回原位不漂移。
    """
    potion = B.Every(cfg.potion_every) if cfg.potion_key else None
    buff = make_buff(cfg)
    round_no = 0
    while True:
        round_no += 1
        maybe_buff(bot, cfg, buff)
        move_secs = B.rnd(cfg.move_secs)
        for side in random_sides():
            B.move(bot, side, move_secs)
            B.settle(bot, cfg.settle_secs)
            times = 1 + (1 if random.random() < cfg.extra_attack_prob else 0)
            for i in range(1, times + 1):
                B.jump_attack(bot, cfg.jump_key, cfg.attack_key,
                              cfg.jump_hold_secs, cfg.jump_rise_secs,
                              cfg.key_hold_secs,
                              f'[第{round_no}轮]{SIDE_CN[side]}跳攻({i}/{times})')
                B.wait(bot, cfg.attack_gap_secs)
                if potion is not None and potion.hit():
                    B.drink_potion(bot, cfg.potion_key, cfg.key_hold_secs,
                                   cfg.potion_pause_secs)
            if random.random() < cfg.hop_prob:
                B.tap(bot, cfg.jump_key, cfg.jump_hold_secs)
                B.wait(bot, cfg.attack_gap_secs)
            B.switch_side(bot, cfg.switch_gap_secs)
        if random.random() < cfg.idle_prob:
            B.wait(bot, cfg.idle_secs)


def _look(bot, finder, cfg, direction, misses):
    """截一张图判断主角在哪半边，返回 (方向, 是否越界, 连续没认到次数)。"""
    bot.gate()                                   # 暂停/失焦时不截图
    frame = V.capture(bot.api.client_rect(bot.hwnd))
    hit = finder.find(frame)
    if hit is None:
        misses += 1
        if misses in (1, 5) or misses % 25 == 0:
            bot.log(f'[截图] 第 {misses} 次没找到「{finder.name}」，沿用上一次方向')
        return direction, False, misses
    left = 0 if cfg.platform_left is None else cfg.platform_left
    right = frame.shape[1] - 1 if cfg.platform_right is None else cfg.platform_right
    chosen, rescue = V.choose_direction(hit.x, left, right, cfg.edge_margin)
    half = '右' if chosen == 'left' else '左'
    turn = '换方向，' if chosen != direction else '继续'
    tail = '（已越过边距，先多挪一段回中间）' if rescue else ''
    bot.log(f'[截图] 名字牌 x={hit.x:.0f}（得分 {hit.score:.2f}）在{half}半边'
            f' → {turn}一直往{SIDE_CN[chosen]}打{tail}')
    return chosen, rescue, 0


def vision_jump(bot, cfg):
    """【截图判断版】每隔 vision_interval 秒截一张图，用名字牌找主角，然后一直朝一个方向打：

    - 主角在右半边就一直往左打，在左半边就一直往右打；跨过中线才换方向，
      不再「左一下右一下」互相抵消，一轮只走一个方向、只打这个方向
    - 每一步都朝远离她所在那一端的方向走，越打越靠中间，不会走出平台掉下去
    - 已经越过安全边距（离端点不足 edge_margin 像素）时按 rescue_scale 倍时长往中间挪，
      挪完立刻重新截图确认，不会一路冲出去
    - 没识别到就沿用上一次方向，不盲改；连续认不到会在日志里提醒
    - 没标定名字牌模板时自动退回 random_jump，标定后下次运行自动生效
    """
    finder = None
    if cfg.vision_enabled and cfg.name_template:
        try:
            finder = V.NameFinder(cfg.name_template, cfg.match_threshold)
        except Exception as error:
            bot.log(f'[提示] 名字牌模板不可用：{error}')
    if finder is None:
        bot.log('[提示] 还没标定名字牌，先按随机版跑；在界面点「截屏标定」后自动切换')
        return random_jump(bot, cfg)

    potion = B.Every(cfg.potion_every) if cfg.potion_key else None
    buff = make_buff(cfg)
    direction, rescue, next_shot, misses, round_no = None, False, 0.0, 0, 0
    while True:
        round_no += 1
        maybe_buff(bot, cfg, buff)
        if time.monotonic() >= next_shot:
            next_shot = time.monotonic() + cfg.vision_interval
            direction, rescue, misses = _look(bot, finder, cfg, direction, misses)
        side = direction or 'right'
        move_secs = B.rnd(cfg.move_secs) * (cfg.rescue_scale if rescue else 1.0)
        if rescue:
            next_shot = 0.0                       # 救援轮之后立刻重新截图确认
        rescue = False
        B.move(bot, side, move_secs)
        B.settle(bot, cfg.settle_secs)
        for i in range(1, cfg.attacks_per_side + 1):
            B.jump_attack(bot, cfg.jump_key, cfg.attack_key,
                          cfg.jump_hold_secs, cfg.jump_rise_secs,
                          cfg.key_hold_secs,
                          f'[第{round_no}轮]{SIDE_CN[side]}跳攻({i}/{cfg.attacks_per_side})')
            B.wait(bot, cfg.attack_gap_secs)
            if potion is not None and potion.hit():
                B.drink_potion(bot, cfg.potion_key, cfg.key_hold_secs,
                               cfg.potion_pause_secs)
        B.wait(bot, cfg.switch_gap_secs)          # 轮间间隔


def static_cast(bot, cfg):
    """【备选积木示范】原地连打，每微调周期左右轻点一次回原位；每 potion_every 下喝一瓶药。

    对应以前“站桩施法 + 反挂载微调”的打法，想用时把 farm.py 的 plan 改成 'static_cast'。
    """
    move_secs = B.rnd(cfg.move_secs)
    cast_count = 0
    potion = B.Every(cfg.potion_every) if cfg.potion_key else None
    buff = make_buff(cfg)
    while True:
        maybe_buff(bot, cfg, buff)
        B.attack_once(bot, cfg.attack_key, cfg.key_hold_secs, f'[施法 {cast_count + 1}]')
        B.wait(bot, cfg.attack_gap_secs)
        cast_count += 1
        if potion is not None and potion.hit():
            B.drink_potion(bot, cfg.potion_key, cfg.key_hold_secs, cfg.potion_pause_secs)
        if cast_count % cfg.micro_move_every == 0:
            B.move(bot, 'right', move_secs)
            B.settle(bot, cfg.settle_secs)
            B.move(bot, 'left', move_secs)
            move_secs = B.rnd(cfg.move_secs)


PLANS = {
    'fixed_jump': fixed_jump,     # 保存版：固定先右后左各一次
    'random_jump': random_jump,   # 随机版：先后手/次数/纯跳/发呆都有随机
    'vision_jump': vision_jump,   # 截图判断版：名字牌定位，偏哪边就先打反方向
    'static_cast': static_cast,   # 站桩施法+微调（示范怎么拼别的打法）
}
