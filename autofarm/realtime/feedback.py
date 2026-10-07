"""Timestamped state history and conservative online result events."""
from collections import deque
from dataclasses import asdict
import json
from pathlib import Path

import cv2

from .control import standing_platform, graph, route
from .model import Actor, Box, Observation, Platform, Rope
from .demo_feedback import VerifiedHUDReader, experience_delta


def observation_data(o):
    return asdict(o) if o is not None else None


def observation_from_data(data):
    if data is None:
        return None
    def actor(d):
        if d is None:
            return None
        return Actor(Box(**d['box']), d['confidence'], d.get('vx', 0), d.get('vy', 0), d.get('track_id', 0))
    d = dict(data)
    d['player'] = actor(d.get('player'))
    d['monsters'] = [actor(a) for a in d.get('monsters', [])]
    d['navigation_targets'] = [actor(a) for a in d.get('navigation_targets', [])]
    d['platforms'] = [Platform(**p) for p in d.get('platforms', [])]
    d['ropes'] = [Rope(**r) for r in d.get('ropes', [])]
    return Observation(**d)


class StateHistory:
    def __init__(self, seconds=2):
        self.seconds = seconds
        self.rows = deque(maxlen=120)
        self.images = deque(maxlen=3)
        self.last_image_at = -float('inf')
        self.last_row_at = -float('inf')
        self.epoch = None

    def reset(self):
        self.rows.clear()
        self.images.clear()
        self.last_image_at = self.last_row_at = -float('inf')

    def observe(self, o, now, offset, previous_input, image=None):
        if o.map_epoch != self.epoch:
            self.reset()
            self.epoch = o.map_epoch
        floor = standing_platform(o) if o.player and not o.reason else None
        p = o.player
        row = dict(t=now, frame_id=o.frame_id, captured_at=o.captured_at,
                   frame_age_ms=(now-o.captured_at)*1000, reason=o.reason,
                   player_world=None if p is None else [p.box.cx-offset[0], p.box.y2-offset[1]],
                   velocity=None if p is None else [p.vx, p.vy], floor_id=floor.id if floor else None,
                   position_quantum=o.position_quantum,
                   position_confidence_kind='quantized_minimap' if o.position_quantum else 'tracker_score',
                   previous_input=previous_input)
        while self.rows and now-self.rows[0]['t'] > self.seconds:
            self.rows.popleft()
        if now-self.last_row_at >= .09:
            self.rows.append(row)
            self.last_row_at = now
        if image is not None and now-self.last_image_at >= .7:
            self.images.append((o.captured_at, image.copy()))
            self.last_image_at = now
        return row

    def snapshot(self, o, now, offset, base, executor, feedback, allow_transfers):
        def actor(a):
            b = a.box
            return dict(id=str(a.track_id), screen_box=[b.x1, b.y1, b.x2, b.y2],
                        world_position=[b.cx-offset[0], b.y2-offset[1]], confidence=a.confidence,
                        vx=a.vx, vy=a.vy)
        floor = standing_platform(o)
        from .farm_plan import suspended_rope_origin
        rope_origin=suspended_rope_origin(o)
        route_origin=floor or (rope_origin[1] if rope_origin else None)
        ps = executor.transfer_platforms(o)
        edges = graph(ps, o.ropes, base.motion) if route_origin and base.motion.calibrated else {}
        destinations = [dict(platform_id=p.id, route=route(edges, route_origin.id, p.id, base.failures),
                        resume_rope_head=route_origin.id if rope_origin else None,
                        can_fire=bool((base.safe_platforms is None or p.id in base.safe_platforms)
                            and (base.firing_platforms is None or p.id in base.firing_platforms)),
                        route_verified=all(tuple(e[:2]) in base.verified_edges
                            for e in route(edges, route_origin.id, p.id, base.failures)),
                        visible_nearby_monsters=sum(bool(abs(m.box.cx-(p.left+p.right)/2)<base.motion.attack_max
                            and -35<m.box.y2-p.y<85) for m in [*o.monsters,*o.navigation_targets]))
                        for p in ps if route_origin and (not floor or p.id != floor.id) and edges
                        and (p.id==route_origin.id or route(edges, route_origin.id, p.id, base.failures))] if allow_transfers else []
        firing_geometry=[]
        margin=executor.clearance(o)
        for perch in o.platforms:
            if base.firing_platforms is not None and perch.id not in base.firing_platforms:continue
            stand_left,stand_right=perch.left+margin,perch.right-margin
            if stand_left>stand_right:continue
            spans=[]
            for target in o.platforms:
                if base.combat_platforms is not None and target.id not in base.combat_platforms:continue
                if target.id==perch.id or not -35<target.y-perch.y<85:continue
                for side,lo,hi in (('left',stand_left-base.motion.attack_max,stand_right-base.motion.attack_min),
                                   ('right',stand_left+base.motion.attack_min,stand_right+base.motion.attack_max)):
                    lo,hi=max(lo,target.left),min(hi,target.right)
                    if hi<=lo:continue
                    spans.append(dict(combat_platform=target.id,direction=side,
                        possible_target_world_x=[float(lo-offset[0]),float(hi-offset[0])],
                        covered_width=float(hi-lo),height_difference=float(target.y-perch.y)))
            firing_geometry.append(dict(platform_id=perch.id,
                allowed_stand_world_x=[float(stand_left-offset[0]),float(stand_right-offset[0])],
                possible_combat_spans=spans,
                semantics='Geometry only: attainable from some permitted stand position, not guaranteed hits or spawns.'))
        return dict(observed_at=o.captured_at, current_time=now, player=actor(o.player),
                    player_floor=floor.id if floor else None, monsters=[actor(m) for m in o.monsters],
                    suspended_rope_origin=None if not rope_origin else dict(head_platform=rope_origin[1].id,
                        attachment_confirmed=False,semantics='Local bounded upward probe must confirm actual attachment before ascent.'),
                    navigation_targets=[actor(m) for m in o.navigation_targets],
                    platforms=[dict(id=p.id, left=p.left-offset[0], right=p.right-offset[0], y=p.y-offset[1])
                               for p in o.platforms],
                    ropes=[dict(x=r.x-offset[0], top=r.top-offset[1], bottom=r.bottom-offset[1]) for r in o.ropes],
                    position_quantum=o.position_quantum, visual_facing=None,
                    submitted_facing=base.applied_facing, motion=asdict(base.motion),
                    task_safe_platforms=sorted(base.safe_platforms) if base.safe_platforms is not None else None,
                    task_firing_platforms=sorted(base.firing_platforms) if base.firing_platforms is not None else None,
                    task_combat_platforms=sorted(base.combat_platforms) if base.combat_platforms is not None else None,
                    task_transit_platforms=sorted(executor.transit_platforms),
                    firing_lanes=base.firing_lanes,
                    available_actions=['wait', 'move', 'face', 'attack','fire']+
                        (['transfer', 'return'] if allow_transfers else []),
                    candidate_destinations=destinations,
                    firing_geometry=firing_geometry,
                    attack_lanes=[dict(id='lane:'+side,direction=side,
                        current_members=[str(m.track_id) for m in executor.attack_members(o,side)],
                        semantics='Attack currently visible, eligible monsters in this fixed direction; stop when none.')
                        for side in ('left','right')],
                    allowed_move_world_x=None if not floor else
                        [float(floor.left-offset[0]+executor.clearance(o)),float(floor.right-offset[0]-executor.clearance(o))],
                    turn_clearance=None if not floor else dict(left=float(o.player.box.cx-floor.left),
                        right=float(floor.right-o.player.box.cx),required=float(executor.turn_clearance(o))),
                    fire_semantics='AI chooses direction and duration; no target identity required. '
                        'Predictive fire may miss; input submission never proves a hit. '
                        'Requires grounded task firing support and no visible close blocker.',
                    action=executor.current.data() if executor.current else None,
                    committed_transfer=executor.busy, history=list(self.rows), feedback=feedback)

    def recent_images(self, now):
        return [image for stamp, image in self.images if 0 <= now-stamp <= self.seconds]

    def policy_images(self, o, now, current):
        frames=[(stamp,image) for stamp,image in self.images
                if 0<=now-stamp<=self.seconds and stamp<o.captured_at]
        if current is not None:frames=frames[-2:]+[(o.captured_at,current)]
        return [im for _,im in frames],[stamp for stamp,_ in frames]


class OnlineHUD:
    """Optional session calibration. Reading happens on the policy worker."""
    def __init__(self, folder):
        folder = Path(folder)
        spec = json.loads((folder/'feedback_config.json').read_text(encoding='utf-8'))
        if spec.get('version') != 1:
            raise ValueError('Invalid feedback configuration')
        self.scene_id = spec['request_id']
        self.region = spec['hud_region']
        self.interval = spec.get('interval', .2)
        self.max_gap = spec.get('max_gap', 2)
        if (not isinstance(self.scene_id, str) or len(self.region) != 4
                or any(type(v) is not int for v in self.region)
                or not 0 <= self.region[0] < self.region[2] <= 8000
                or not 0 <= self.region[1] < self.region[3] <= 8000
                or type(self.interval) not in (float, int) or not .1 <= self.interval <= 2
                or type(self.max_gap) not in (float, int) or not .5 <= self.max_gap <= 5):
            raise ValueError('Invalid feedback sampling parameters')
        path = (folder/spec['calibration']).resolve()
        if not path.is_relative_to(folder.resolve()):
            raise ValueError('Feedback calibration escapes session')
        self.reader = VerifiedHUDReader(path.parent, json.loads(path.read_text(encoding='utf-8')))
        self.anchor_first_level = spec.get('anchor_level_from_first_frame', False)
        if type(self.anchor_first_level) is not bool:
            raise ValueError('Invalid level reference switch')

    def read(self, image):
        x1, y1, x2, y2 = self.region
        hud = image[y1:y2, x1:x2]
        if self.anchor_first_level and hud.shape == self.reader.shape:
            a,b,c,d = self.reader.config['level_roi']
            self.reader.level_reference = hud[b:d,a:c].copy()
            self.reader.level_label = None
            self.anchor_first_level = False
        return self.reader.read(hud)


class FeedbackTracker:
    def __init__(self, max_gap=2):
        self.events = deque(maxlen=120)
        self.max_gap = max_gap
        self.epoch = None
        self.previous_floor = None
        self.floor_since = None
        self.last_floor = None
        self.targets = set()
        self.last_hp = None
        self.candidate = self.confirmed = None
        self.reward_samples = deque(maxlen=600)
        self.floor_reward_start=None;self.floor_attack_seconds=0.;self.last_observed_at=None
        self.exp = dict(exp=None, level=None, reason='not_calibrated')
        self.moving_at = None
        self.moving_origin = None
        self.stuck_reported = False

    def emit(self, kind, now, o=None, **data):
        event = dict(event=kind, t=now, frame_id=o.frame_id if o else None, **data)
        self.events.append(event)
        return event

    def invalidate(self):
        self.exp = dict(exp=None, level=None, reason='observation_interrupted')
        self.previous_floor = self.last_floor = self.floor_since = None
        self.targets.clear()
        self.candidate = self.confirmed = None
        self.reward_samples.clear()
        self.floor_reward_start=None;self.floor_attack_seconds=0.;self.last_observed_at=None
        self.last_hp = None
        self.moving_at = self.moving_origin = None
        self.stuck_reported = False

    def observe(self, o, now, offset=(0, 0), previous_input=None, health=None):
        produced = []
        if o is None or o.reason or not o.player or not o.motion_valid:
            self.invalidate()
            return produced
        if o.map_epoch != self.epoch:
            self.invalidate()
            self.epoch = o.map_epoch
        floor = standing_platform(o)
        ident = floor.id if floor else None
        changed_floor=ident!=self.previous_floor
        if ident != self.previous_floor:
            self.previous_floor, self.floor_since = ident, now
            self.floor_attack_seconds=0.
            self.floor_reward_start=self.confirmed if ident and self.confirmed and 0<=now-self.confirmed['t']<=self.max_gap else None
        if ident and ident != self.last_floor and self.floor_since is not None and now-self.floor_since >= .15:
            produced.append(self.emit('landing_observed', now, o, platform=ident))
            self.last_floor = ident
        targets = {str(m.track_id) for m in o.monsters}
        for lost in sorted(self.targets-targets):
            produced.append(self.emit('target_lost', now, o, target=lost, kill=None, damage=None))
        self.targets = targets
        hp = health.get('hp') if health else None
        if hp is not None and self.last_hp is not None and hp < self.last_hp:
            produced.append(self.emit('hp_decreased', now, o, delta=hp-self.last_hp,
                                      attribution='unknown', death=None))
        self.last_hp = hp
        keys = set(previous_input.get('held_keys', [])) if previous_input else set()
        if ident and not changed_floor and 'shift' in keys and self.last_observed_at is not None:
            dt=now-self.last_observed_at
            if 0<=dt<=.1:self.floor_attack_seconds+=dt
        self.last_observed_at=now
        moving = bool(keys.intersection({'left', 'right'})) and floor is not None
        position = (o.player.box.cx-offset[0], o.player.box.y2-offset[1])
        if not moving:
            self.moving_at = self.moving_origin = None
            self.stuck_reported = False
        elif self.moving_at is None:
            self.moving_at, self.moving_origin = now, position
        elif abs(position[0]-self.moving_origin[0])+abs(position[1]-self.moving_origin[1]) > max(12, o.position_quantum):
            self.moving_at, self.moving_origin = now, position
            self.stuck_reported = False
        elif now-self.moving_at > 1 and not self.stuck_reported:
            produced.append(self.emit('movement_no_observed_progress', now, o,
                                      seconds=now-self.moving_at, position_quantum=o.position_quantum))
            self.stuck_reported = True
        return produced

    def hud(self, reading, sampled_at, now, o=None):
        """Require repeated readings; gaps, rollovers and missing glyphs stay unknown."""
        row = dict(reading, t=sampled_at)
        self.exp = row
        if row.get('exp') is None or row.get('level') is None:
            self.candidate = self.confirmed = None
            self.reward_samples.clear()
            self.floor_reward_start=None
            return [self.emit('experience_unknown', now, o, reading=row)]
        previous = self.candidate
        self.candidate = row
        if (previous is None or not 0 < row['t']-previous['t'] <= self.max_gap
                or row['exp'] != previous['exp'] or row['level'] != previous['level']):
            if previous and (row['level'] != previous['level'] or row['t']-previous['t'] > self.max_gap):
                self.confirmed = None
                self.reward_samples.clear()
                self.floor_reward_start=None
            return []
        before = self.confirmed
        self.confirmed = row
        if self.previous_floor and self.floor_reward_start is None:self.floor_reward_start=row
        if self.reward_samples and (row['level']!=self.reward_samples[-1]['level']
                or row['t']-self.reward_samples[-1]['t']>self.max_gap
                or row['exp']<self.reward_samples[-1]['exp']):
            self.reward_samples.clear()
        self.reward_samples.append(row)
        while self.reward_samples and row['t']-self.reward_samples[0]['t']>60:
            self.reward_samples.popleft()
        if before is None:
            return [self.emit('experience_baseline_confirmed', now, o, reading=row)]
        delta = experience_delta(before, row, max_gap=self.max_gap)
        if delta.get('net_exp') is None:
            return [self.emit('experience_unknown', now, o, reading=row, reason=delta['reason'])]
        if delta['net_exp']:
            return [self.emit('experience_changed', now, o, **delta, reading=row)]
        return []

    def status(self, now):
        confirmed=self.confirmed if self.confirmed and 0<=now-self.confirmed['t']<=self.max_gap else None
        current_confirmed=bool(confirmed and self.exp.get('t')==confirmed['t'])
        floor_net=None
        if (confirmed and self.floor_reward_start and confirmed['level']==self.floor_reward_start['level']
                and confirmed['exp']>=self.floor_reward_start['exp']):
            floor_net=confirmed['exp']-self.floor_reward_start['exp']
        window=None
        if confirmed and len(self.reward_samples)>1:
            first=self.reward_samples[0];seconds=confirmed['t']-first['t']
            if seconds>0:
                net=confirmed['exp']-first['exp']
                window=dict(observed_seconds=seconds,net_exp=net,exp_per_minute=net*60/seconds,
                            evidence='continuous repeated calibrated readings',target_exp_per_minute=1633.6)
        return dict(experience=dict(self.exp,confirmed=current_confirmed),
                    confirmed_experience=confirmed,
                    experience_window=window,
                    standing_observation=dict(platform=self.previous_floor,
                        seconds_here=now-self.floor_since if self.floor_since is not None else None,
                        attack_input_seconds=self.floor_attack_seconds,net_exp_here=floor_net,
                        attribution='observed interval only; individual hits and kills unknown'),
                    recent_events=[e for e in self.events if 0 <= now-e['t'] <= 2],
                    damage=None, kills=None, visual_facing=None)
