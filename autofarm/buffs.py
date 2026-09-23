"""Shared Buff scheduling; only the plan's action thread sends keys."""
from dataclasses import dataclass
import math
import random
import time
import queue

from .winapi import VK_CODES


@dataclass(frozen=True)
class BuffSlot:
    key: str
    interval: tuple
    pause: tuple


def parse_slots(rows, forbidden=()):
    slots = []
    for row in rows:
        key = str(row.get('key', '')).strip().lower()
        if not key:
            continue
        if key not in VK_CODES or key in {'left', 'right', 'up', 'down', 'f11', 'f12', *forbidden}:
            raise ValueError(f'Buff 按键不可用或与动作冲突：{key}')
        interval = (float(row['lo']), float(row['hi']))
        pause = (float(row['pmin']), float(row['pmax']))
        for name, pair, positive in [('间隔', interval, True), ('停顿', pause, False)]:
            if (not all(math.isfinite(v) for v in pair) or pair[0] > pair[1]
                    or pair[0] < 0 or (positive and pair[0] == 0)):
                raise ValueError(f'Buff {key} 的{name}必须是有效的递增秒数区间')
        slots.append(BuffSlot(key, interval, pause))
    return tuple(slots)


class BuffScheduler:
    def __init__(self, cfg):
        slots = getattr(cfg, 'buff_slots', None)
        if slots is None:
            slots = (BuffSlot(cfg.buff_key, cfg.buff_every_secs, cfg.buff_pause_secs),) if cfg.buff_key else ()
        self.slots = slots
        self.hold = getattr(cfg, 'buff_hold_secs', .12)
        immediate = getattr(cfg, 'buff_start_immediately', False)
        self.next_at = [time.monotonic() + (0 if immediate else random.uniform(*s.interval)) for s in slots]
        self.ready_at = 0.0
        self.release_at = None

    def sync(self, bot):
        commands = getattr(bot, 'buff_commands', None)
        if isinstance(commands, queue.Queue):
            while True:
                try:
                    kind, slots = commands.get_nowait()
                except queue.Empty:
                    break
                if kind == 'replace':
                    old = dict(zip(self.slots, self.next_at))
                    self.slots = slots
                    self.next_at = [old.get(s, time.monotonic()) for s in slots]
                    self.release_at = None
                    bot.log('[Buff] 设置已应用：' + (', '.join(s.key for s in slots) or '已关闭'))
                elif kind == 'now' and self.slots:
                    self.next_at = [time.monotonic()] * len(self.slots)
                    bot.log('[Buff] 已安排各技能补一次')
                elif kind == 'hold':
                    self.hold = slots
        self.report(bot)

    def report(self, bot, reason=''):
        if not self.slots:
            bot.buff_status = 'Buff 已关闭'
        else:
            now = time.monotonic()
            bot.buff_status = ' | '.join(f'{s.key}: {max(0, at-now):.0f}秒' for s, at in zip(self.slots, self.next_at))
            if reason:
                bot.buff_status += '；' + reason

    def prepare(self, now):
        """Attack must have been released for 150ms; callers keep scanning meanwhile."""
        if self.release_at is None:
            self.release_at = now
        return now - self.release_at >= .15

    def cancel_prepare(self):
        self.release_at = None

    def cooling(self):
        return bool(self.ready_at and time.monotonic() < self.ready_at)

    def due(self):
        return not self.cooling() and any(time.monotonic() >= at for at in self.next_at)

    def cast_one(self, bot, wait=True):
        if self.cooling():
            return False
        for i in sorted(range(len(self.slots)), key=self.next_at.__getitem__):
            slot = self.slots[i]
            if time.monotonic() < self.next_at[i]:
                continue
            # A pause/focus change must not queue a stale key for later delivery.
            if not bot.try_tap(slot.key, self.hold):
                self.report(bot, '暂停或失焦，尚未发送完成')
                return False
            now = time.monotonic()
            self.next_at[i] = now + random.uniform(*slot.interval)
            pause = random.uniform(*slot.pause)
            self.ready_at = now + pause
            self.release_at = None
            bot.log(f'[Buff] 已发送 {slot.key}（{self.hold*1000:.0f}ms）；下次约 {self.next_at[i] - now:.0f} 秒后')
            self.report(bot, '按键已发送')
            if wait:
                bot.wait(pause)
            return True
        return False


def run(bot, cfg):
    scheduler = BuffScheduler(cfg)
    if not scheduler.slots:
        raise ValueError('仅定时 Buff 模式至少需要一个 Buff 槽位')
    while True:
        bot.gate()
        scheduler.sync(bot)
        scheduler.cast_one(bot)
        bot.wait(.05)
