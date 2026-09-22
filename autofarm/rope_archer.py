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
    def __init__(self, profile, player_name=''):
        path = V.asset_path(profile)
        self.data = json.loads(path.read_text(encoding='utf-8'))
        d = self.data
        d.setdefault('key_lease_secs', .4)
        d.setdefault('move_stop_tolerance', 6)
        d.setdefault('position_uncertainty', 4)
        d.setdefault('release_latency_secs', .02)
        d.setdefault('edge_margin', 18)
        d.setdefault('max_frame_age', .12)
        d.setdefault('movement_floor_tolerance', 4)
        d.setdefault('nameplate_offset_y', 20)
        self.typed_name = None
        self.allow_player_images = True
        if player_name.strip():
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
                   'player_y', 'floor_tolerance', 'max_speed', 'move_stop_tolerance',
                   'face_hold', 'key_lease_secs', 'scan_interval', 'position_uncertainty',
                   'release_latency_secs', 'edge_margin', 'max_frame_age', 'movement_floor_tolerance',
                   'nameplate_offset_y')
        if any(not math.isfinite(d[k]) for k in numeric):
            raise ValueError('射手参数必须是有限数值')
        if not (d['safe_left'] < d['target_x'] - d['target_tolerance'] <
                d['target_x'] + d['target_tolerance'] < d['safe_right']):
            raise ValueError('回位区必须完整位于安全边界内')
        if not (0 < d['move_stop_tolerance'] < d['target_tolerance'] and
                0 < d['face_hold'] <= .06 and .2 <= d['key_lease_secs'] <= .6 and
                d['max_speed'] > 0 and d['floor_tolerance'] > 0 and
                .03 <= d['scan_interval'] <= .5):
            raise ValueError('射手动作时长或移动速度参数无效')
        reserve = d['position_uncertainty'] + d['release_latency_secs'] * d['max_speed']
        if not (d['position_uncertainty'] >= 3 and .01 <= d['release_latency_secs'] <= .1
                and d['edge_margin'] >= 15 and .04 <= d['max_frame_age'] <= .15
                and 0 < d['movement_floor_tolerance'] <= d['floor_tolerance']
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
        if self.typed_name:
            box = list(d['player_roi'])
            name_y = d['player_y'] - d['nameplate_offset_y']
            box[1], box[3] = name_y - 28, name_y + 28
            name_region = self.region(frame, a, box)
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
            if self.typed_name:
                return Observation(anchor=a, reason=f'未认到输入角色名「{self.typed_name.name}」或对应备用图，停止动作')
            return Observation(anchor=a, reason=f'未认到人物名字牌（最高 {best_score:.3f}，'
                               f'阈值 {d["player_threshold"]:.3f}；局部片段不足），停止动作')
        p = max(candidates, key=lambda h: h.score)
        if self.typed_name and not source.startswith('名字'):
            source += '（图片备用）'
        if abs(p.y - a.y - d['player_y']) > d['floor_tolerance']:
            return Observation(p, a, reason=f'人物高度异常（相对高度 {p.y - a.y:.1f}，'
                               f'预期 {d["player_y"]:.1f}），停止动作（可能掉层或被击飞）')
        if not d['safe_left'] <= p.x - a.x <= d['safe_right']:
            return Observation(p, a, reason='人物超出已标定的安全平台，停止动作')
        box = list(d['monkey_roi'])
        box[0] = max(box[0], p.x - a.x + 25)
        region = self.region(frame, a, box)
        if region is None:
            return Observation(p, a, reason='猴子观察区域超出窗口，停止动作')
        hits = [h for f in self.monkeys if (h := self.find_in(f, region)) is not None]
        monkey = max(hits, key=lambda h: h.score) if hits else None
        monkey_source = '头部/身体' if monkey else ''
        if monkey is None and self.halo:
            halo_box = list(d['monkey_halo_roi'])
            halo_box[0] = max(halo_box[0], p.x - a.x + 25)
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
    """带停稳范围的回位状态机；返回期望保持的按键，不逐帧重复点按。"""
    def __init__(self, data):
        self.d = data
        self.reset()

    def reset(self):
        self.facing_right = False
        self.previous = None
        self.moving = None
        self.face_until = 0.0
        self.settle_until = 0.0

    def _move(self, x):
        d = self.d
        # 即使下一帧完全卡住，也只走到回位区以内；预扣定位误差和松键延迟。
        limit = (d['target_x'] + d['target_tolerance'] if self.moving == 'right'
                 else d['target_x'] - d['target_tolerance'])
        room = (limit - x if self.moving == 'right' else x - limit) - self.reserve
        lease = min(d['key_lease_secs'], room / d['max_speed'])
        if lease <= .005:
            return None
        self.facing_right = self.moving == 'right'
        return self.moving, lease, '持续按住方向键回位'

    @property
    def reserve(self):
        return self.d['position_uncertainty'] + self.d['release_latency_secs'] * self.d['max_speed']

    def decide(self, o, now):
        d = self.d
        if o.reason or not o.player or not o.anchor:
            self.reset()
            return None
        x, y = o.player.x - o.anchor.x, o.player.y - o.anchor.y
        if not d['safe_left'] <= x <= d['safe_right'] or abs(y - d['player_y']) > d['floor_tolerance']:
            self.reset()
            return None
        previous, self.previous = self.previous, (x, y)
        # 启动/失焦恢复需要两帧确认；正常移动不再因位移超过 9px 而断续。
        if (previous is None or abs(y - previous[1]) > 3 or
                abs(y - d['player_y']) > d['movement_floor_tolerance']):
            self.moving = None
            self.facing_right = False
            self.face_until = 0.0
            return None
        goal = d['target_x'] - d['move_stop_tolerance']
        # 进入右侧缓冲带时，回到内侧优先于攻击、转向和到位后的等待。
        if x > d['target_x'] + d['target_tolerance']:
            self.moving = 'left'
            self.face_until = 0.0
            return self._move(x)
        if self.moving:
            arrived = x >= goal if self.moving == 'right' else x <= goal
            if not arrived:
                return self._move(x)
            self.moving = None
            self.settle_until = now + .15
            return None
        if now < self.settle_until:
            return None
        if abs(x - d['target_x']) > d['target_tolerance']:
            self.moving = 'right' if x < d['target_x'] else 'left'
            return self._move(x)
        if o.monkey is None:
            return None
        if not self.facing_right:
            if x + d['face_hold'] * d['max_speed'] + self.reserve > d['target_x'] + d['target_tolerance']:
                self.moving = 'left'
                return self._move(x)
            self.facing_right = True
            self.face_until = now + d['face_hold']
            return 'right', d['face_hold'], '转向右侧'
        if now < self.face_until:
            return 'right', self.face_until - now, '转向右侧'
        return 'shift', d['key_lease_secs'], '右侧有猴子，持续按住 Shift'


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
                'player' if reason.startswith(('未认到人物名字牌', '未认到输入角色名', '未认到唯一角色名')) else
                'height' if reason.startswith('人物高度异常') else 'position')
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
    if not cfg.vision_enabled:
        raise ValueError('绳边射手必须启用截图识别，不支持纯计时攻击')
    scene = RopeScene(cfg.archer_profile, cfg.archer_player_name)
    controller = ArcherController(scene.data)
    bot.log(f'[绳边射手 v{ARCHER_VERSION}] 内侧站位 / 光圈识别；有猴子长按 Shift，无怪松开')
    if scene.typed_name:
        bot.log(f'[定位模式] 输入角色名：{scene.typed_name.name}；本地 OCR + 对应名字图片备用')
    diagnostics = FailureSnapshots(V.program_dir() / 'captures' / 'diagnostics')
    last_reason = last_action = last_source = None
    with HeldInput(bot) as inputs:
        while True:
            if bot.gate():
                inputs.clear()
                controller.reset()
            epoch = inputs.epoch
            key_at_capture = inputs.key
            started = time.monotonic()
            frame = V.capture(bot.api.client_rect(bot.hwnd))
            o = scene.observe(frame)
            now = time.monotonic()
            if now - started > scene.data['max_frame_age'] or epoch != inputs.epoch:
                inputs.clear()
                controller.reset()
                bot.wait(scene.data['scan_interval'])
                continue
            reason = o.reason or (f'右侧发现猴子（{o.monkey_source}）' if o.monkey else '右侧无猴子，等待')
            if reason != last_reason:
                bot.log(f'[识别] {reason}')
                last_reason = reason
            if o.player_source and o.player_source != last_source:
                bot.log(f'[定位] {o.player_source}')
            last_source = o.player_source
            action = controller.decide(o, now)
            if action:
                key, lease, label = action
                deadline = now + lease
                # 已经按着移动键时，截图/识别期间也在移动，须从截图时算剩余边距。
                if key in ('left', 'right') and key_at_capture == key:
                    deadline = started + lease
                if inputs.apply(key, deadline, epoch):
                    if label != last_action:
                        bot.log(f'[动作] {label}')
                    last_action = label
                else:
                    controller.reset()
                    last_action = None
            else:
                inputs.clear()
                if last_action:
                    bot.log('[动作] 松开按键')
                last_action = None
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
