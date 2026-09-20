"""Pure decision logic. Scene pixels and minimap pixels never mix."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Entity:
    x: float
    y: float
    score: float = 1.0


@dataclass(frozen=True)
class Action:
    keys: tuple = ()
    duration: float = 0.0
    reason: str = "idle"
    skill: str | None = None
    recovery: float = 0.0


class Planner:
    def __init__(self, config):
        self.skills = sorted(config['profiles'][config['profile']]['skills'],
                             key=lambda s: s.get('priority', 0), reverse=True)
        self.nav = config['navigation']
        self.ready = {}
        self.facing = None
        self.busy_until = 0
        self.waypoint = 0
        self.anchor = None
        self.anchor_time = 0
        self.blocked = False

    def reset(self):
        self.facing = None
        self.anchor = None
        self.blocked = False

    def commit(self, action, now):
        if action.keys in [('left',), ('right',)]:
            self.facing = action.keys[0]
        if action.skill:
            skill = next(s for s in self.skills if s['name'] == action.skill)
            self.ready[action.skill] = now + skill['cooldown']
            self.busy_until = now + action.duration + action.recovery

    def decide(self, player, monsters, minimap_player, now):
        if player is None:
            self.anchor = None
            return Action(reason='player not detected')
        if self.blocked:
            return Action(reason='stuck: press F8 twice to reset')
        if now < self.busy_until:
            return Action(reason='skill recovery')
        monsters = sorted(monsters, key=lambda m: math.hypot(m.x-player.x, m.y-player.y))
        # A ready skill can select any target it reaches, not only the nearest.
        for skill in self.skills:
            if now < self.ready.get(skill['name'], 0):
                continue
            for target in monsters:
                if abs(target.x-player.x) <= skill['range_x'] and abs(target.y-player.y) <= skill['range_y']:
                    facing = 'right' if target.x >= player.x else 'left'
                    if skill.get('directional', True) and self.facing != facing:
                        return Action((facing,), 0.05, 'face target')
                    self.anchor = None
                    return Action((skill['key'],), skill['hold'], 'attack', skill['name'], skill.get('recovery', 0))
        # Wait for cooldown instead of walking into a target already in range.
        if any(abs(m.x-player.x) <= s['range_x'] and abs(m.y-player.y) <= s['range_y']
               for m in monsters for s in self.skills):
            self.anchor = None
            return Action(reason='cooldown')
        same_floor = [m for m in monsters if abs(m.y-player.y) <= self.nav['same_floor_tolerance']]
        if same_floor:
            keys = ('right',) if same_floor[0].x > player.x else ('left',)
        else:
            route = self.nav.get('waypoints', [])
            if not route or minimap_player is None:
                self.anchor = None
                return Action(reason='no reachable target / route')
            point = route[self.waypoint]
            dx, dy = point['x']-minimap_player.x, point['y']-minimap_player.y
            tolerance = self.nav['waypoint_tolerance']
            if abs(dx) <= tolerance and abs(dy) <= tolerance:
                self.waypoint = (self.waypoint+1) % len(route)
                self.anchor = None
                return Action(reason='waypoint reached')
            keys = (('right',) if dx > 0 else ('left',)) if abs(dx) > tolerance else tuple(point.get('vertical_keys', []))
            if not keys:
                return Action(reason='vertical route needs configured keys')
        position = minimap_player or player
        if self.anchor is None or math.hypot(position.x-self.anchor.x, position.y-self.anchor.y) > 2:
            self.anchor, self.anchor_time = position, now
        elif now-self.anchor_time > self.nav['stuck_seconds']:
            self.blocked = True
            return Action(reason='stuck: press F8 twice to reset')
        return Action(keys, self.nav['move_pulse'], 'move')
