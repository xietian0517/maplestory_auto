"""Strict scene interchange. Coordinates are pixels of the submitted image."""
from dataclasses import dataclass, field
import math


def number(value, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Expected finite number')
    if not low <= value <= high:
        raise ValueError('Coordinate/parameter outside allowed range')
    return float(value)


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self): return (self.x1 + self.x2) / 2
    @property
    def cy(self): return (self.y1 + self.y2) / 2
    @property
    def width(self): return self.x2 - self.x1
    @property
    def height(self): return self.y2 - self.y1

    def moved(self, dx, dy):
        return Box(self.x1+dx, self.y1+dy, self.x2+dx, self.y2+dy)

    @classmethod
    def parse(cls, values, width, height):
        if not isinstance(values, list) or len(values) != 4:
            raise ValueError('Box must contain four coordinates')
        b = cls(*(number(v, 0, width if i % 2 == 0 else height) for i, v in enumerate(values)))
        if b.width < 3 or b.height < 3:
            raise ValueError('Empty or tiny box')
        return b


@dataclass(frozen=True)
class Platform:
    id: str
    left: float
    right: float
    y: float


@dataclass(frozen=True)
class Rope:
    x: float
    top: float
    bottom: float


@dataclass
class Scene:
    request_id: str
    map_name: str
    width: int
    height: int
    play_area: Box
    name_box: Box
    foot_offset: float
    monsters: list
    exclusions: list
    platforms: list
    ropes: list
    preferred: list
    confidence: float
    reasoning: str
    source: str = 'external_gpt'

    def to_data(self):
        def box(b): return [b.x1,b.y1,b.x2,b.y2]
        return dict(request_id=self.request_id,map_name=self.map_name,width=self.width,height=self.height,
                    play_area=box(self.play_area),player_name_box=box(self.name_box),player_foot_offset=self.foot_offset,
                    monster_boxes=[box(b) for b in self.monsters],exclude_boxes=[box(b) for b in self.exclusions],
                    platforms=[vars(p) for p in self.platforms],ropes=[vars(r) for r in self.ropes],
                    preferred_platforms=self.preferred,confidence=self.confidence,reasoning=self.reasoning)

    @classmethod
    def parse(cls, data, request_id, width, height):
        if not isinstance(data, dict) or data.get('request_id') != request_id:
            raise ValueError('Scene belongs to another request')
        if data.get('width') != width or data.get('height') != height:
            raise ValueError('Scene image size mismatch')
        area = Box.parse(data['play_area'], width, height)
        name = Box.parse(data['player_name_box'], width, height)
        offset = number(data['player_foot_offset'], -80, 20)
        monsters = data['monster_boxes']
        if not isinstance(monsters, list) or len(monsters) > 12:
            raise ValueError('At most 12 monster examples')
        monsters = [Box.parse(x, width, height) for x in monsters]
        exclusions = data['exclude_boxes']
        if not isinstance(exclusions, list) or len(exclusions) > 30:
            raise ValueError('Too many exclusions')
        exclusions = [Box.parse(x, width, height) for x in exclusions]
        for b in [name, *monsters]:
            if b.x1 < area.x1 or b.x2 > area.x2 or b.y1 < area.y1 or b.y2 > area.y2:
                raise ValueError('Entity outside play area')
        platforms, ids = [], set()
        if not isinstance(data['platforms'], list) or not 1 <= len(data['platforms']) <= 80:
            raise ValueError('Expected 1–80 platforms')
        for p in data['platforms']:
            ident = p['id']
            if not isinstance(ident, str) or not 1 <= len(ident) <= 40 or ident in ids:
                raise ValueError('Invalid/duplicate platform id')
            left, right = number(p['left'], 0, width), number(p['right'], 0, width)
            if right-left < 15: raise ValueError('Platform too narrow')
            platforms.append(Platform(ident, left, right, number(p['y'], 0, height)))
            ids.add(ident)
        ropes = []
        if not isinstance(data['ropes'], list) or len(data['ropes']) > 40:
            raise ValueError('Too many ropes')
        for r in data['ropes']:
            rope = Rope(number(r['x'], 0, width), number(r['top'], 0, height), number(r['bottom'], 0, height))
            if rope.bottom <= rope.top: raise ValueError('Invalid rope')
            ropes.append(rope)
        preferred = data['preferred_platforms']
        if not isinstance(preferred, list) or len(preferred) > 80 or any(p not in ids for p in preferred):
            raise ValueError('Unknown preferred platform')
        if not isinstance(data['map_name'], str) or len(data['map_name']) > 100:
            raise ValueError('Invalid map name')
        if not isinstance(data['reasoning'], str) or len(data['reasoning']) > 2000:
            raise ValueError('Invalid explanation')
        return cls(request_id, data['map_name'], width, height, area, name, offset,
                   monsters, exclusions, platforms, ropes, preferred,
                   number(data['confidence'], 0, 1), data['reasoning'])


@dataclass(frozen=True)
class Actor:
    box: Box
    confidence: float
    vx: float = 0
    vy: float = 0
    track_id: int = 0


@dataclass
class Observation:
    frame_id: int
    captured_at: float
    player: Actor | None = None
    monsters: list = field(default_factory=list)
    platforms: list = field(default_factory=list)
    ropes: list = field(default_factory=list)
    reason: str = ''
    map_epoch: int = 0
    motion_valid: bool = True
    navigation_targets: list = field(default_factory=list)


@dataclass(frozen=True)
class Decision:
    keys: frozenset = frozenset()
    reason: str = 'wait'
    target: str = ''


@dataclass
class MotionProfile:
    speed: float = 220
    jump_height: float = 0
    jump_distance: float = 0
    calibrated: bool = False
    attack_min: float = 65
    attack_max: float = 400
    edge_margin: float = 28

    @classmethod
    def parse(cls, d):
        if type(d.get('calibrated')) is not bool: raise ValueError('calibrated must be boolean')
        return cls(number(d['speed'], 50, 600), number(d['jump_height'], 0, 250),
                   number(d['jump_distance'], 0, 600), d['calibrated'],
                   number(d.get('attack_min', 65), 30, 200),
                   number(d.get('attack_max', 400), 200, 650),
                   number(d.get('edge_margin', 28), 20, 100))
