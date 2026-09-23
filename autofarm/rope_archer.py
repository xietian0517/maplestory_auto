"""绳边射手：持续持键回位，右侧有猴子时持续按住 Shift。"""
from dataclasses import dataclass
import json
import math
import time

import cv2
import numpy as np

from . import vision as V
from .version import ARCHER_VERSION


@dataclass(frozen=True)
class Observation:
    player: object = None
    anchor: object = None
    monkey: object = None
    reason: str = ''
    player_source: str = ''
    monkey_source: str = ''


class HaloFinder(V.NameFinder):
    """只比较浅黄色发光椭圆，背景和被钱币遮住的身体不参与匹配。"""
    @staticmethod
    def mask(frame):
        b, g, r = cv2.split(frame.astype(np.int16))
        return ((r >= 205) & (g >= 205) & (b >= 105) &
                (g - b >= 8) & (np.abs(r - g) <= 35)).astype(np.uint8) * 255

    def __init__(self, path, threshold):
        super().__init__(path, threshold)
        self.template = self.mask(self.template)
        if self.template.std() < 5:
            raise ValueError('光圈模板缺少浅黄色椭圆特征')

    def best(self, frame):
        return super().best(self.mask(frame))


class RopeScene:
    def __init__(self, profile, player_name='', name_template='', template_owner=''):
        path = V.asset_path(profile)
        self.data = json.loads(path.read_text(encoding='utf-8'))
        d = self.data
        d.setdefault('key_lease_secs', .4)
        d.setdefault('move_stop_tolerance', 6)
        d.setdefault('position_uncertainty', 4)
        d.setdefault('release_latency_secs', .02)
        d.setdefault('edge_margin', 18)
        d.setdefault('max_frame_age', .12)
        d.setdefault('nameplate_offset_y', 20)
        d.setdefault('attack_range', 450)
        d.setdefault('face_confirm_distance', 6)
        self.typed_name = None
        self.custom_player = None
        self.allow_player_images = True
        if name_template:
            from .custom_template import CustomNameFinder, check_owner
            check_owner(player_name, template_owner)
            self.custom_player = CustomNameFinder(name_template)
            self.allow_player_images = False
        elif player_name.strip():
            from .name_ocr import TypedNameFinder, normalize_name
            self.typed_name = TypedNameFinder(player_name)
            # 现成称号图仅供模板中绑定的角色使用，换名字不继承别人的图。
            self.allow_player_images = normalize_name(player_name) == normalize_name(d.get('player_name', ''))
        self.anchor = V.NameFinder(path.parent / d['anchor_template'], d['anchor_threshold'])
        self.last_anchor = None
        self.player = V.NameFinder(path.parent / d['player_template'], d['player_threshold'])
        self.players = [(self.player, d['player_offset_x'])]
        for variant in d.get('player_variants', []):
            self.players.append((V.NameFinder(path.parent / variant['template'], d['player_threshold']),
                                 variant['offset_x']))
        partial_threshold = d.get('player_partial_threshold', .90)
        minimum_parts = d.get('player_partial_min_parts', 2)
        if not (isinstance(partial_threshold, (int, float)) and .86 <= partial_threshold <= 1
                and type(minimum_parts) is int and 2 <= minimum_parts <= 4):
            raise ValueError('局部称号识别需要阈值至少 0.86、至少两个片段')
        self.player_parts = [(V.PartialNameFinder(finder, partial_threshold, minimum_parts), offset)
                             for finder, offset in self.players]
        self.monkeys = [V.NameFinder(path.parent / p, d['monkey_threshold'])
                        for p in d['monkey_templates']]
        if not self.monkeys:
            raise ValueError('射手模板必须包含猴子图片')
        for finder in self.monkeys[:]:
            # 两种朝向都识别；只翻转识别模板，不改变原文件。
            import copy
            flipped = copy.copy(finder)
            flipped.template = cv2.flip(finder.template, 1)
            self.monkeys.append(flipped)
        self.halo = (HaloFinder(path.parent / d['monkey_halo_template'], d['monkey_halo_threshold'])
                     if d.get('monkey_halo_template') else None)
        numeric = ('safe_left', 'safe_right', 'target_x', 'target_tolerance',
                   'player_y', 'max_speed', 'move_stop_tolerance',
                   'face_hold', 'key_lease_secs', 'scan_interval', 'position_uncertainty',
                   'release_latency_secs', 'edge_margin', 'max_frame_age',
                   'nameplate_offset_y', 'attack_range', 'face_confirm_distance')
        if any(not math.isfinite(d[k]) for k in numeric):
            raise ValueError('射手参数必须是有限数值')
        if not (d['safe_left'] < d['target_x'] - d['target_tolerance'] <
                d['target_x'] + d['target_tolerance'] < d['safe_right']):
            raise ValueError('回位区必须完整位于安全边界内')
        if not (0 < d['move_stop_tolerance'] < d['target_tolerance'] and
                0 < d['face_hold'] <= .2 and .2 <= d['key_lease_secs'] <= .6 and
                d['max_speed'] > 0 and
                .03 <= d['scan_interval'] <= .5):
            raise ValueError('射手动作时长或移动速度参数无效')
        if not (50 <= d['attack_range'] <= 600 and 6 <= d['face_confirm_distance'] <= 10):
            raise ValueError('射程或转向确认距离无效')
        reserve = d['position_uncertainty'] + d['release_latency_secs'] * d['max_speed']
        if not (d['position_uncertainty'] >= 3 and .01 <= d['release_latency_secs'] <= .1
                and d['edge_margin'] >= 15 and .04 <= d['max_frame_age'] <= .15
                and reserve < d['target_tolerance'] - d['move_stop_tolerance']
                and d['safe_left'] + d['edge_margin'] < d['target_x'] - d['target_tolerance']
                and d['target_x'] + d['target_tolerance'] <= d['safe_right'] - d['edge_margin']):
            raise ValueError('防掉落余量不足：回位区、松键延迟和位置误差需要同时留白')

    @staticmethod
    def region(frame, anchor, box):
        x1, y1, x2, y2 = [int(round(v + (anchor.x if i % 2 == 0 else anchor.y)))
                          for i, v in enumerate(box)]
        if not (0 <= x1 < x2 <= frame.shape[1] and 0 <= y1 < y2 <= frame.shape[0]):
            return None
        return frame[y1:y2, x1:x2], x1, y1

    @staticmethod
    def find_in(finder, region, **kwargs):
        if region is None:
            return None
        image, x, y = region
        hit = finder.find(image, **kwargs)
        return V.Hit(hit.x + x, hit.y + y, hit.score) if hit else None

    def _find_anchor(self, frame):
        # 连续运行优先在上一帧平台附近搜索，减少持键期间的识别延迟。
        if self.last_anchor is not None:
            h, w = self.anchor.template.shape[:2]
            a = self.last_anchor
            x, y = max(0, int(a.x - w / 2 - 80)), max(0, int(a.y - h / 2 - 80))
            right, bottom = min(frame.shape[1], int(a.x + w / 2 + 80)), min(frame.shape[0], int(a.y + h / 2 + 80))
            if right > x and bottom > y:
                hit = self.anchor.find(frame[y:bottom, x:right])
                if hit:
                    self.last_anchor = V.Hit(hit.x + x, hit.y + y, hit.score)
                    return self.last_anchor
        a = self.anchor.best(frame)
        self.last_anchor = a if a and a.score >= self.anchor.threshold else None
        return a

    def observe(self, frame):
        a = self._find_anchor(frame)
        if a is None or a.score < self.anchor.threshold:
            score = a.score if a else 0.0
            return Observation(reason=f'未认到绳边平台（最高 {score:.3f}，'
                               f'阈值 {self.anchor.threshold:.3f}），停止动作')
        d = self.data
        region = self.region(frame, a, d['player_roi'])
        candidates, best_score = [], 0.0
        source = '完整称号牌'
        if self.custom_player:
            hit = self.find_in(self.custom_player, region)
            if self.custom_player.ambiguous:
                return Observation(anchor=a, reason='未认到唯一人物名字图片（有多个相同候选），停止动作')
            if hit:
                candidates.append(V.Hit(hit.x, hit.y + d['nameplate_offset_y'], hit.score))
                source = '粘贴名字图片'
        if self.typed_name:
            # OCR 与名字图片共用人物搜索区域，不再限定标定高度附近的窄条。
            name_region = region
            # 首帧先读文字；之后若现成备用图可用，避免反复慢 OCR 打断连续按键。
            initial_read = self.typed_name.last_ocr == -float('inf')
            hit = self.find_in(self.typed_name, name_region,
                               allow_ocr=initial_read or not self.allow_player_images)
            if self.typed_name.ambiguous:
                return Observation(anchor=a, reason='未认到唯一角色名（有多个相同候选），停止动作')
            if hit:
                candidates.append(V.Hit(hit.x, hit.y + d['nameplate_offset_y'], hit.score))
                source = self.typed_name.source
        if not candidates and region is not None and self.allow_player_images:
            image, rx, ry = region
            for finder, offset in self.players:
                hit = finder.best(image)
                if hit is not None:
                    best_score = max(best_score, hit.score)
                    if hit.score >= finder.threshold:
                        candidates.append(V.Hit(hit.x + rx + offset, hit.y + ry, hit.score))
        if not candidates and region is not None and self.allow_player_images:
            # 优先原生像素版本；仍只使用这一帧的可见片段，不沿用历史坐标。
            for finder, offset in self.player_parts:
                partial = finder.find(image)
                if partial:
                    hit, count = partial
                    candidates.append(V.Hit(hit.x + rx + offset, hit.y + ry, hit.score))
                    source = f'称号局部 {count}/4'
                    break
        if not candidates and self.typed_name and self.allow_player_images and not initial_read:
            hit = self.find_in(self.typed_name, name_region)
            if self.typed_name.ambiguous:
                return Observation(anchor=a, reason='未认到唯一角色名（有多个相同候选），停止动作')
            if hit:
                candidates.append(V.Hit(hit.x, hit.y + d['nameplate_offset_y'], hit.score))
                source = self.typed_name.source
        if not candidates:
            if self.custom_player:
                return Observation(anchor=a, reason='未认到粘贴的名字图片，停止动作；请检查图片是否为当前角色的原尺寸完整名字')
            if self.typed_name:
                return Observation(anchor=a, reason=f'未认到输入角色名「{self.typed_name.name}」或对应备用图，停止动作')
            return Observation(anchor=a, reason=f'未认到人物名字牌（最高 {best_score:.3f}，'
                               f'阈值 {d["player_threshold"]:.3f}；局部片段不足），停止动作')
        p = max(candidates, key=lambda h: h.score)
        if self.typed_name and not source.startswith('名字'):
            source += '（图片备用）'
        if not d['safe_left'] <= p.x - a.x <= d['safe_right']:
            return Observation(p, a, reason='人物超出已标定的安全平台，停止动作')
        box = list(d['monkey_roi'])
        box[0] = max(box[0], p.x - a.x + 25)
        # 在射程内搜索，避免远处高分目标盖过附近可攻击的猴子。
        box[2] = min(box[2], p.x - a.x + d['attack_range'])
        region = self.region(frame, a, box)
        if region is None:
            return Observation(p, a, reason='猴子观察区域超出窗口，停止动作')
        hits = [h for f in self.monkeys if (h := self.find_in(f, region)) is not None]
        monkey = max(hits, key=lambda h: h.score) if hits else None
        monkey_source = '头部/身体' if monkey else ''
        if monkey is None and self.halo:
            halo_box = list(d['monkey_halo_roi'])
            halo_box[0] = max(halo_box[0], p.x - a.x + 25)
            halo_box[2] = min(halo_box[2], p.x - a.x + d['attack_range'])
            monkey = self.find_in(self.halo, self.region(frame, a, halo_box))
            if monkey:
                monkey_source = '头顶光圈'
        return Observation(p, a, monkey, player_source=source, monkey_source=monkey_source)

    def annotate(self, frame, observation):
        o, d = observation, self.data
        view = V.annotate(frame, o.player, o.reason and 'STOP' or
                          ('MONKEY: HOLD Shift' if o.monkey else 'WAIT: no monkey'))
        if o.anchor:
            a = o.anchor
            y = round(a.y + d['player_y'])
            for key, color in [('safe_left', (0, 0, 255)), ('safe_right', (0, 0, 255)),
                               ('target_x', (255, 255, 0))]:
                x = round(a.x + d[key])
                cv2.line(view, (x, y - 95), (x, y + 20), color, 2)
            # 橙线为主动移动上限，比红色平台边界更靠内。
            x = round(a.x + d['target_x'] + d['target_tolerance'])
            cv2.line(view, (x, y - 95), (x, y + 20), (0, 165, 255), 2)
            r = self.region(frame, a, d['monkey_roi'])
            if r:
                im, x, yy = r
                cv2.rectangle(view, (x, yy), (x + im.shape[1], yy + im.shape[0]), (255, 255, 0), 1)
        if o.monkey:
            cv2.circle(view, (round(o.monkey.x), round(o.monkey.y)), 25, (0, 255, 255), 2)
            if o.monkey_source == '头顶光圈':
                cv2.putText(view, 'MONKEY: halo', (10, 85), cv2.FONT_HERSHEY_SIMPLEX,
                            .65, (0, 255, 255), 2)
        if o.player_source.startswith('称号局部'):
            cv2.putText(view, 'PLAYER: partial badge', (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        .65, (0, 255, 255), 2)
        return view


class ArcherController:
    """先保护边缘，再清怪；右键生效须由当前画面的实际位移确认。"""
    def __init__(self, data):
        self.d = data
        self.reset()

    def reset(self):
        self.facing_right = False
        self.observed_frames = 0
        self.moving = None
        self.settle_until = 0.0
        self.turn_origin = None
        self.last_right_at = -float('inf')
        self.wait_reason = ''

    @property
    def reserve(self):
        return self.d['position_uncertainty'] + self.d['release_latency_secs'] * self.d['max_speed']

    def applied(self, key, o, now):
        """仅在持键层成功提交后记账；计划发右键不等于转向成功。"""
        if key == 'right' and not self.facing_right:
            if self.turn_origin is None or now - self.last_right_at > .25:
                self.turn_origin = (o.player.x - o.anchor.x, o.player_source)
            self.last_right_at = now
        elif key == 'left':
            self._forget_facing()

    def _forget_facing(self):
        self.facing_right = False
        self.turn_origin = None
        self.last_right_at = -float('inf')

    def _confirm_turn(self, o, x, now):
        origin = self.turn_origin
        if self.facing_right or origin is None:
            return False
        ox, source = origin
        if (now - self.last_right_at > .25 or source != o.player_source or
                x < ox - 1):
            self.turn_origin = None
            return False
        if x - ox >= self.d['face_confirm_distance']:
            self.facing_right = True
            self.turn_origin = None
            self.settle_until = now + .08
            return True
        return False

    def _move(self, x):
        d = self.d
        limit = (d['target_x'] + d['target_tolerance'] if self.moving == 'right'
                 else d['target_x'] - d['target_tolerance'])
        room = (limit - x if self.moving == 'right' else x - limit) - self.reserve
        goal = d['target_x'] - d['move_stop_tolerance']
        remaining = goal - x if self.moving == 'right' else x - goal
        # 按停点限制持键时长；平台边界只作额外上限，不能充当回位目的地。
        lease = min(d['key_lease_secs'], room / d['max_speed'], max(0, remaining) / d['max_speed'])
        # 向左回位从计划开始就撤销攻击许可，即使按键随后提交失败。
        if self.moving == 'left':
            self._forget_facing()
        if lease <= .005:
            self.wait_reason = '移动余量不足，松键重新确认位置'
            return None
        return self.moving, lease, '持续按住方向键回位'

    def motion_stopped(self, now):
        self.moving = None
        self.settle_until = max(self.settle_until, now + .15)
        self.wait_reason = '已进入停靠范围，松键等待；范围内不追着坐标微调'

    def _turn(self, x):
        d = self.d
        room = d['target_x'] + d['target_tolerance'] - x - self.reserve
        progress = max(0, x - self.turn_origin[0]) if self.turn_origin else 0
        needed = max(1, d['face_confirm_distance'] - progress)
        if room < needed:
            self.moving = 'left'
            return self._move(x)
        turn_distance = needed + d['position_uncertainty']
        return 'right', min(d['face_hold'], room / d['max_speed'], turn_distance / d['max_speed']), '向右转向，等待实际右移确认（未确认不攻击）'

    def can_buff(self, o, now, action):
        if o.reason or not o.player or not o.anchor or self.observed_frames < 2 or self.moving:
            return False
        d = self.d
        x = o.player.x - o.anchor.x
        return (d['safe_left'] + d['edge_margin'] <= x <= d['target_x'] + d['target_tolerance']
                and now >= self.settle_until and self.turn_origin is None
                and (action is None or action[0] == 'shift'))

    def decide(self, o, now):
        d = self.d
        self.wait_reason = ''
        if o.reason or not o.player or not o.anchor:
            self.reset()
            return None
        x = o.player.x - o.anchor.x
        if not d['safe_left'] <= x <= d['safe_right']:
            self.reset()
            return None
        self.observed_frames = min(2, self.observed_frames + 1)
        if self.observed_frames < 2:
            self.wait_reason = '等待下一帧确认人物位置'
            return None
        just_confirmed = self._confirm_turn(o, x, now)
        goal = d['target_x'] - d['move_stop_tolerance']
        # 两端缓冲区先保位置；右侧退回过程必须完整结束，不能中途反复转向。
        if x > d['target_x'] + d['target_tolerance']:
            self.moving = 'left'
            return self._move(x)
        if x < d['safe_left'] + d['edge_margin']:
            self.moving = 'right'
            return self._move(x)
        if self.moving == 'left':
            if x > goal + d['move_stop_tolerance']:
                return self._move(x)
            self.motion_stopped(now)
            return None
        in_range = (o.monkey is not None and
                    25 <= o.monkey.x - o.player.x <= d['attack_range'])
        if in_range:
            # 安全区内优先打得到的怪，不为恢复固定站位打断战斗。
            was_moving = self.moving is not None
            self.moving = None
            if just_confirmed or was_moving:
                self.settle_until = max(self.settle_until, now + .08)
                self.wait_reason = '停止移动，等待站稳后向右攻击'
                return None
            if now < self.settle_until:
                self.wait_reason = '等待移动结束后站稳'
                return None
            if not self.facing_right:
                return self._turn(x)
            return 'shift', d['key_lease_secs'], '右朝向已确认，优先清理射程内猴子'
        # 无目标立即停攻；等待本身不改变朝向，不能因此制造下一轮左右转向。
        if self.moving:
            if x < goal - d['move_stop_tolerance']:
                return self._move(x)
            self.motion_stopped(now)
            return None
        if now < self.settle_until:
            return None
        if abs(x - d['target_x']) > d['target_tolerance']:
            self.moving = 'right' if x < d['target_x'] else 'left'
            return self._move(x)
        self.wait_reason = '右侧射程内无猴子，等待'
        return None


class FailureSnapshots:
    """松键后按类型保留最新原图；每类最多五秒写一次，文件数量固定。"""
    def __init__(self, folder):
        self.folder = folder
        self.saved_at = {}

    def record(self, frame, observation, now):
        reason = observation.reason
        if not reason:
            return None
        kind = ('anchor' if reason.startswith('未认到绳边平台') else
                'player' if reason.startswith(('未认到人物名字牌', '未认到输入角色名', '未认到唯一', '未认到粘贴')) else
                'position')
        if now - self.saved_at.get(kind, -float('inf')) < 5:
            return None
        self.saved_at[kind] = now
        self.folder.mkdir(parents=True, exist_ok=True)
        out = self.folder / f'last_{kind}_failure.png'
        cv2.imencode('.png', frame)[1].tofile(str(out))
        out.with_suffix('.json').write_text(json.dumps({
            'version': ARCHER_VERSION, 'reason': reason,
            'player': vars(observation.player) if observation.player else None,
            'anchor': vars(observation.anchor) if observation.anchor else None,
        }, ensure_ascii=False, indent=2), encoding='utf-8')
        return out


def run(bot, cfg):
    from .held_input import HeldInput
    from .buffs import BuffScheduler
    if not cfg.vision_enabled:
        raise ValueError('绳边射手必须启用截图识别，不支持纯计时攻击')
    scene = RopeScene(cfg.archer_profile, cfg.archer_player_name,
                      cfg.archer_name_template, cfg.archer_template_owner)
    controller = ArcherController(scene.data)
    buffs = BuffScheduler(cfg)
    bot.log(f'[绳边射手 v{ARCHER_VERSION}] 确认右转后攻击 / 安全区先清怪后回位 / 光圈识别')
    if scene.custom_player:
        bot.log('[定位模式] 使用玩家粘贴的名字图片；匹配失败即停止动作')
    elif scene.typed_name:
        bot.log(f'[定位模式] 输入角色名：{scene.typed_name.name}；本地 OCR + 对应名字图片备用')
    diagnostics = FailureSnapshots(V.program_dir() / 'captures' / 'diagnostics')
    last_reason = last_action = last_source = last_wait = None
    with HeldInput(bot) as inputs:
        while True:
            if bot.gate():
                inputs.clear()
                controller.reset()
            buffs.sync(bot)
            epoch = inputs.epoch
            key_at_capture = inputs.key
            started = time.monotonic()
            frame = V.capture(bot.api.client_rect(bot.hwnd))
            o = scene.observe(frame)
            now = time.monotonic()
            if now - started > scene.data['max_frame_age'] or epoch != inputs.epoch:
                inputs.clear()
                controller.reset()
                if last_wait != 'slow':
                    bot.log('[等待] 截图识别超时或前台状态变化，松键重新识别')
                last_wait = 'slow'
                bot.wait(scene.data['scan_interval'])
                continue
            reason = o.reason or (f'右侧发现猴子（{o.monkey_source}）' if o.monkey else '右侧射程内无猴子，等待')
            if reason != last_reason:
                bot.log(f'[识别] {reason}')
                last_reason = reason
            if o.player_source and o.player_source != last_source:
                bot.log(f'[定位] {o.player_source}')
            last_source = o.player_source
            action = controller.decide(o, now)
            if buffs.cooling():
                # Keep observing during the animation; no direction/attack overlaps.
                inputs.clear()
                bot.wait(scene.data['scan_interval'])
                continue
            if buffs.due() and controller.can_buff(o, now, action):
                inputs.clear()
                if not buffs.prepare(now):
                    buffs.report(bot, '已松开攻击，等待技能间隔')
                    bot.wait(scene.data['scan_interval'])
                    continue
                cast = buffs.cast_one(bot, wait=False)
                if cast and epoch == inputs.epoch and not bot.paused and bot.foreground():
                    # 施放 Buff 不改变方向；下一帧重新检查位置，无须再左右挪动。
                    controller.motion_stopped(time.monotonic())
                else:
                    controller.reset()
                last_action = None
                continue  # Always use a new screenshot after the Buff.
            buffs.cancel_prepare()
            if buffs.due():
                buffs.report(bot, '已到期，等待人物识别和安全站位')
            if action:
                key, lease, label = action
                deadline = now + lease
                # 已经按着移动键时，截图/识别期间也在移动，须从截图时算剩余边距。
                if key in ('left', 'right') and key_at_capture == key:
                    deadline = started + lease
                if inputs.apply(key, deadline, epoch):
                    controller.applied(key, o, now)
                    last_wait = None
                    if label != last_action:
                        bot.log(f'[动作] {label}')
                    last_action = label
                else:
                    if epoch != inputs.epoch or bot.paused or not bot.foreground():
                        controller.reset()
                    else:
                        # 截图期间短移动租期已耗尽：松键等新位置，不把朝向一并清掉。
                        controller.motion_stopped(time.monotonic())
                    last_action = None
            else:
                inputs.clear()
                if last_action:
                    bot.log('[动作] 松开按键')
                last_action = None
                if controller.wait_reason and controller.wait_reason != last_wait:
                    bot.log(f'[等待] {controller.wait_reason}')
                last_wait = controller.wait_reason
                if o.reason:
                    try:
                        saved = diagnostics.record(frame, o, now)
                        if saved:
                            bot.log(f'[诊断原图] {saved}')
                    except (OSError, cv2.error) as error:
                        bot.log(f'[诊断] 原图保存失败：{error}')
            if bot.wait(scene.data['scan_interval']):
                inputs.clear()
                controller.reset()
