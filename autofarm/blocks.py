"""小功能积木：每一块都是一个简单动作，plans 里随便拼。

积木约定：第一个参数是 bot，时间参数传 (最小, 最大) 秒的二元组，运行时随机取值。
以后你用文字描述新打法，直接在这里选积木拼到 plans.py 里即可。
"""
import random
import time

DIRECTION_CN = {'right': '右', 'left': '左', 'up': '上', 'down': '下'}


def rnd(rng):
    """(min, max) 之间随机一个秒数。"""
    return random.uniform(rng[0], rng[1])


class Every:
    """每 n 次触发一次的计数器，例如喝药。"""

    def __init__(self, n):
        self.n = n
        self.count = 0

    def hit(self):
        self.count += 1
        return self.count % self.n == 0


class Interval:
    """每隔一段时间触发一次（时长随机），用于加 Buff 这类定时动作。

    首次到点 = 创建后 rng 秒；之后每次触发自动排下一次随机间隔。
    循环里每轮问一句 interval.due() 即可，问得勤也不会误触发。
    """

    def __init__(self, rng):
        self.rng = rng
        self.next_at = time.monotonic() + rnd(rng)

    def due(self):
        """到点返回 True 并自动排下一次；没到点返回 False。"""
        now = time.monotonic()
        if now >= self.next_at:
            self.next_at = now + rnd(self.rng)
            return True
        return False


# ---------- 基础积木 ----------

def tap(bot, key, hold_rng):
    """点按一个键（攻击/技能/喝药通用）。"""
    bot.gate()
    bot.tap(key, rnd(hold_rng))


def wait(bot, rng):
    """随机等待（可被 F12/失焦打断）。"""
    bot.wait(rnd(rng))


def move(bot, direction, seconds):
    """朝一个方向移动固定秒数（左右配对时请在外层用同一个 seconds，保证回原位）。"""
    bot.gate()
    bot.log(f'[移动] {DIRECTION_CN.get(direction, direction)} {seconds * 1000:.0f}ms')
    bot.hold_key(direction, seconds)


def settle(bot, rng):
    """移动后停稳等待，避免立刻反打导致位移吃不满。"""
    bot.wait(rnd(rng))


def switch_side(bot, rng):
    """换方向间隔（最终版 250~450ms），防止方向键冲突，看着也自然。"""
    bot.wait(rnd(rng))


def attack_once(bot, key, hold_rng, label=''):
    """原地打一下。"""
    bot.gate()
    if label:
        bot.log(label)
    bot.tap(key, rnd(hold_rng))


def jump_attack(bot, jump_key, attack_key, jump_hold_rng, rise_rng,
                attack_hold_rng, label=''):
    """跳起来打：按跳 -> 等腾空 -> 按攻击。跳跃到出招这半秒一气呵成，中途不响应暂停。"""
    bot.gate()
    if label:
        bot.log(label)
    bot.tap(jump_key, rnd(jump_hold_rng))       # Alt 起跳
    time.sleep(rnd(rise_rng))                   # 等角色腾空
    if not bot.foreground():
        return                                   # 起跳后失焦就放弃这一击，不乱发键
    bot.down(attack_key)
    time.sleep(rnd(attack_hold_rng))
    bot.up(attack_key)


def burst(bot, key, times, gap_rng, hold_rng, side_cn=''):
    """原地连打 times 下，每下之间随机间隔。返回期间不会移动。"""
    for i in range(1, times + 1):
        attack_once(bot, key, hold_rng,
                   f'[{side_cn}攻({i}/{times})]' if side_cn else f'[攻击({i}/{times})]')
        if i < times:
            wait(bot, gap_rng)


def drink_potion(bot, key, hold_rng, pause_rng, name='回蓝'):
    """喝一瓶药，喝完停一下。"""
    bot.gate()
    bot.log(f'[{name}] 喝药1瓶')
    bot.tap(key, rnd(hold_rng))
    wait(bot, pause_rng)


def cast_buff(bot, key, hold_rng, pause_rng, name='加Buff'):
    """按一下 Buff 键（默认 Home），按完停一下等技能动作播完。"""
    bot.gate()
    bot.log(f'[{name}] 按 {key}')
    bot.tap(key, rnd(hold_rng))
    wait(bot, pause_rng)
