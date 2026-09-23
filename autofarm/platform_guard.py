"""User calibrated platform guarding, with anchor relative geometry and two directions."""
from dataclasses import dataclass, replace
import copy
import json
import math
from pathlib import Path
import time

import cv2

from . import vision as V
from .custom_template import CustomNameFinder
from .name_ocr import TypedNameFinder, normalize_name
from .rope_archer import ArcherController, Observation, RopeScene


def validate_profile(d):
    required = ('safe_left', 'safe_right', 'target_x', 'target_tolerance', 'max_speed')
    if any(not isinstance(d.get(k), (int, float)) or not math.isfinite(d[k]) for k in required):
        raise ValueError('守台方案缺少有效的平台 / 站位参数，请重新标定')
    if not (100 <= d['max_speed'] <= 600 and d['target_tolerance'] >= 20
            and d['target_tolerance'] > 10 + .02*d['max_speed']
            and d['safe_left'] + 20 < d['target_x'] - d['target_tolerance']
            and d['target_x'] + d['target_tolerance'] < d['safe_right'] - 20):
        raise ValueError('停靠范围宽度至少 40px，且左右各距平台边界超过 20px')
    for key in ('player_roi', 'monster_roi'):
        box = d.get(key)
        if (not isinstance(box, list) or len(box) != 4 or
                not all(isinstance(v, (int, float)) and math.isfinite(v) for v in box) or
                box[0] >= box[2] or box[1] >= box[3]):
            raise ValueError(f'{key} 区域无效')
    for key in ('anchor_threshold', 'monster_threshold'):
        if not isinstance(d.get(key), (int, float)) or not .75 <= d[key] <= 1:
            raise ValueError(f'{key} 必须在 0.75~1 之间')
    if not isinstance(d.get('monster_templates'), list) or not 1 <= len(d['monster_templates']) <= 24:
        raise ValueError('需要 1~24 张怪物模板')
    for filename in [d.get('anchor_template'), d.get('player_template'), *d['monster_templates']]:
        if not isinstance(filename, str) or not filename:
            raise ValueError('模板文件名无效')
    return d


def profile_asset(profile_path, filename):
    parent = profile_path.parent.resolve()
    path = (parent / filename).resolve()
    if not path.is_relative_to(parent):
        raise ValueError('方案图片必须位于方案文件夹内')
    return path


@dataclass(frozen=True)
class GuardObservation:
    player: object = None
    anchor: object = None
    left: object = None
    right: object = None
    reason: str = ''
    player_source: str = ''


class GuardScene:
    _find_anchor = RopeScene._find_anchor

    def __init__(self, profile, player_name='', attack_range=450):
        if not profile:
            raise ValueError('请先在自定义守台页标定或载入方案')
        path = V.asset_path(profile)
        self.data = validate_profile(json.loads(path.read_text(encoding='utf-8')))
        d = self.data
        self.anchor = V.NameFinder(profile_asset(path, d['anchor_template']), d['anchor_threshold'])
        self.last_anchor = None
        self.player = CustomNameFinder(profile_asset(path, d['player_template']))
        self.typed = TypedNameFinder(player_name) if player_name.strip() else None
        self.fallback = not self.typed or normalize_name(player_name) == normalize_name(d.get('player_name', ''))
        self.attack_range = attack_range
        self.monsters = []
        for filename in d['monster_templates']:
            finder = V.NameFinder(profile_asset(path, filename), d['monster_threshold'])
            self.monsters.append(finder)
            flipped = copy.copy(finder)
            flipped.template = cv2.flip(finder.template, 1)
            self.monsters.append(flipped)

    @staticmethod
    def region(frame, anchor, box):
        x1, y1, x2, y2 = [round(v + (anchor.x if i % 2 == 0 else anchor.y)) for i, v in enumerate(box)]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        return (frame[y1:y2, x1:x2], x1, y1) if x1 < x2 and y1 < y2 else None

    def observe(self, frame):
        a = self._find_anchor(frame)
        if not a or a.score < self.anchor.threshold:
            return GuardObservation(reason='未找到标定的固定参照物，停止动作')
        region = self.region(frame, a, self.data['player_roi'])
        p, source = None, ''
        if self.typed:
            p = RopeScene.find_in(self.typed, region)
            if self.typed.ambiguous:
                return GuardObservation(anchor=a, reason='人物名字不唯一，停止动作')
            source = self.typed.source
        if p is None and self.fallback:
            p = RopeScene.find_in(self.player, region)
            source = '标定名字图片'
        if not p:
            return GuardObservation(anchor=a, reason='未找到唯一的人物名字，停止动作')
        if not self.data['safe_left'] <= p.x - a.x <= self.data['safe_right']:
            return GuardObservation(p, a, reason='人物超出左右平台边界，停止动作')
        found = {}
        for side in ('left', 'right'):
            box = list(self.data['monster_roi'])
            low, high = ((p.x - self.attack_range, p.x - 25) if side == 'left'
                         else (p.x + 25, p.x + self.attack_range))
            box[0], box[2] = max(box[0], low-a.x), min(box[2], high-a.x)
            region = self.region(frame, a, box)
            hits = [h for f in self.monsters if (h := RopeScene.find_in(f, region)) is not None]
            found[side] = min(hits, key=lambda h: abs(h.x-p.x)) if hits else None
        return GuardObservation(p, a, found['left'], found['right'], player_source=source)

    def annotate(self, frame, o):
        view = V.annotate(frame, o.player, 'STOP' if o.reason else 'CUSTOM PLATFORM GUARD')
        if o.anchor:
            for key, color in [('safe_left', (0, 0, 255)), ('safe_right', (0, 0, 255)), ('target_x', (255, 255, 0))]:
                x = round(o.anchor.x + self.data[key])
                cv2.line(view, (x, 0), (x, view.shape[0]), color, 2)
            region = self.region(frame, o.anchor, self.data['monster_roi'])
            if region:
                im, x, y = region
                cv2.rectangle(view, (x, y), (x+im.shape[1], y+im.shape[0]), (255, 255, 0), 2)
        for hit in (o.left, o.right):
            if hit:
                cv2.circle(view, (round(hit.x), round(hit.y)), 24, (0, 255, 255), 2)
        return view


class GuardController:
    """Reuse tested horizontal recovery in a mirrored coordinate system for left attacks."""
    def __init__(self, data, direction='right', attack_key='shift', attack_range=450):
        from .winapi import VK_CODES
        if direction not in ('left', 'right', 'both'):
            raise ValueError('攻击方向必须为左、右或两侧')
        if attack_key not in VK_CODES or attack_key in ('left', 'right', 'up', 'down', 'f11', 'f12'):
            raise ValueError('攻击键不能是移动键或启停热键')
        self.data, self.direction, self.attack_key, self.attack_range = data, direction, attack_key, attack_range
        self.side = 'left' if direction == 'left' else 'right'
        self.missing_since = None
        self._create()

    def _create(self):
        d = self.data
        left, right, target = d['safe_left'], d['safe_right'], d['target_x']
        if self.side == 'left':
            left, right, target = -right, -left, -target
        self.inner = ArcherController(dict(safe_left=left, safe_right=right, target_x=target,
            target_tolerance=d['target_tolerance'], move_stop_tolerance=6, max_speed=d['max_speed'],
            position_uncertainty=4, release_latency_secs=.02, edge_margin=18,
            key_lease_secs=.4, face_hold=.12, face_confirm_distance=6, attack_range=self.attack_range))

    def reset(self):
        self.inner.reset()
        self.missing_since = None

    def transformed(self, o):
        sign = -1 if self.side == 'left' else 1
        def hit(h):
            return replace(h, x=h.x*sign) if h else None
        return Observation(hit(o.player), hit(o.anchor), hit(getattr(o, self.side)), o.reason, o.player_source, '用户怪物模板')

    def key(self, key):
        if key == 'shift':
            return self.attack_key
        if self.side == 'left' and key in ('left', 'right'):
            return 'right' if key == 'left' else 'left'
        return key

    def decide(self, o, now):
        if self.direction == 'both' and not o.reason and o.player:
            current = getattr(o, self.side)
            other = 'left' if self.side == 'right' else 'right'
            if current:
                self.missing_since = None
            elif getattr(o, other):
                if self.missing_since is None:
                    self.missing_since = now
                if now - self.missing_since >= .3:
                    self.side = other
                    self._create()
                    self.missing_since = None
            else:
                self.missing_since = None
        action = self.inner.decide(self.transformed(o), now)
        if not action:
            return None
        key, lease, label = action
        label = ('向左攻击目标' if self.side == 'left' else '向右攻击目标') if key == 'shift' else '守台回位 / 确认朝向'
        return self.key(key), lease, label

    def applied(self, key, o, now):
        if key in ('left', 'right'):
            self.inner.applied(self.key(key), self.transformed(o), now)

    def can_buff(self, o, now, action):
        logical = ('shift', 0, '') if action and action[0] == self.attack_key else action
        return self.inner.can_buff(self.transformed(o), now, logical)

    def motion_stopped(self, now):
        self.inner.motion_stopped(now)


def run(bot, cfg):
    from .held_input import HeldInput
    from .buffs import BuffScheduler
    scene = GuardScene(cfg.guard_profile, cfg.guard_player_name, cfg.guard_attack_range)
    controller = GuardController(scene.data, cfg.guard_direction, cfg.guard_attack_key, cfg.guard_attack_range)
    buffs = BuffScheduler(cfg)
    bot.log(f'[自定义守台] {scene.data.get("name", "自定义平台")}；方向={cfg.guard_direction}；攻击键={cfg.guard_attack_key}')
    last = None
    with HeldInput(bot) as inputs:
        while True:
            if bot.gate():
                inputs.clear()
                controller.reset()
            buffs.sync(bot)
            epoch, held, started = inputs.epoch, inputs.key, time.monotonic()
            frame = V.capture(bot.api.client_rect(bot.hwnd))
            o = scene.observe(frame)
            now = time.monotonic()
            if now-started > .12 or epoch != inputs.epoch:
                inputs.clear()
                controller.reset()
                if last != 'slow':
                    bot.log('[守台等待] 识别耗时超过 120ms 或前台状态变化，松键重新识别')
                last = 'slow'
                bot.wait(.04)
                continue
            state = o.reason or f'左侧{"有怪" if o.left else "无怪"} / 右侧{"有怪" if o.right else "无怪"}'
            if state != last:
                bot.log('[守台识别] ' + state)
                last = state
            action = controller.decide(o, now)
            if buffs.cooling():
                inputs.clear()
                bot.wait(.04)
                continue
            if buffs.due() and controller.can_buff(o, now, action):
                inputs.clear()
                if buffs.prepare(now):
                    if buffs.cast_one(bot, wait=False) and epoch == inputs.epoch:
                        controller.motion_stopped(time.monotonic())
                    else:
                        controller.reset()
                else:
                    buffs.report(bot, '已松开攻击，等待技能间隔')
                bot.wait(.04)
                continue
            buffs.cancel_prepare()
            if buffs.due():
                buffs.report(bot, '等待有效识别和安全站位')
            if action:
                key, lease, _ = action
                deadline = (started if held == key and key in ('left', 'right') else now) + lease
                if inputs.apply(key, deadline, epoch):
                    controller.applied(key, o, now)
                elif epoch != inputs.epoch or bot.paused or not bot.foreground():
                    controller.reset()
                else:
                    controller.motion_stopped(time.monotonic())
            else:
                inputs.clear()
            if bot.wait(.04):
                inputs.clear()
                controller.reset()
