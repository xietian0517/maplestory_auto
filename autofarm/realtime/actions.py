"""Policy intents and observed executors. This module never sends input."""
from dataclasses import dataclass, asdict
import math

from .model import Decision
from .control import standing_platform, graph, route


ACTIONS = ('wait', 'move', 'face', 'attack', 'fire', 'transfer', 'return', 'farm')


def duration_limit(action):
    return 120 if action == 'farm' else 20 if action == 'fire' else 3


def finite(value, low, high):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError('Invalid action number')
    return float(value)


@dataclass(frozen=True)
class ActionIntent:
    action_id: str
    source: str
    request_id: str
    frame_id: int
    map_epoch: int
    scene_id: str
    issued_at: float
    valid_until: float
    action: str
    target: str = ''
    world_x: float | None = None
    direction: str = ''
    duration: float = .5
    reason: str = ''

    def __post_init__(self):
        if self.source != 'ai' or self.action not in ACTIONS:
            raise ValueError('Invalid action source or kind')
        for name in ('action_id','request_id','scene_id'):
            value=getattr(self,name)
            if not isinstance(value,str) or not value or len(value)>100:
                raise ValueError('Invalid action identity')
        if type(self.frame_id) is not int or self.frame_id<0 or type(self.map_epoch) is not int or self.map_epoch<0:
            raise ValueError('Invalid action observation identity')
        finite(self.issued_at,0,1e15);finite(self.valid_until,0,1e15)
        if self.valid_until<=self.issued_at:raise ValueError('Invalid action validity interval')
        finite(self.duration,.05,duration_limit(self.action))
        if self.world_x is not None:finite(self.world_x,-20000,20000)
        if self.direction not in ('','left','right'):raise ValueError('Invalid direction')
        if not isinstance(self.target,str) or len(self.target)>80 or not isinstance(self.reason,str) or len(self.reason)>500:
            raise ValueError('Invalid action target or explanation')
        if self.action=='move' and self.world_x is None:raise ValueError('Move requires a world position')
        if self.action in ('face','fire') and not self.direction:raise ValueError('Action requires a direction')
        if self.action=='fire' and (self.target or self.world_x is not None):raise ValueError('Fire uses direction only')
        if self.action=='farm' and (not self.target or self.world_x is None or not self.direction):
            raise ValueError('Farm requires platform, stand position and direction')
        if self.action in ('attack','transfer','return') and not self.target:raise ValueError('Action requires a target')

    def data(self):
        return asdict(self)

    @classmethod
    def from_reply(cls, reply, request):
        """Bind provenance to the actual request, never to model-supplied times."""
        expected = {'request_id', 'frame_id', 'map_epoch', 'action', 'target', 'world_x',
                    'direction', 'duration', 'reason'}
        if not isinstance(reply, dict) or set(reply) != expected:
            raise ValueError('Invalid action response fields')
        if (reply['request_id'] != request.request_id or type(reply['frame_id']) is not int
                or reply['frame_id'] != request.frame_id or type(reply['map_epoch']) is not int
                or reply['map_epoch'] != request.map_epoch):
            raise ValueError('Action response belongs to another observation')
        if reply['action'] not in ACTIONS:
            raise ValueError('Unknown action')
        if not isinstance(reply['target'], str) or len(reply['target']) > 80:
            raise ValueError('Invalid action target')
        if reply['direction'] not in ('', 'left', 'right'):
            raise ValueError('Invalid action direction')
        if not isinstance(reply['reason'], str) or len(reply['reason']) > 500:
            raise ValueError('Invalid action explanation')
        duration = finite(reply['duration'], .05, duration_limit(reply['action']))
        x = None if reply['world_x'] is None else finite(reply['world_x'], -20000, 20000)
        if reply['action'] == 'move' and x is None:
            raise ValueError('Move requires a world position')
        if reply['action'] == 'face' and not reply['direction']:
            raise ValueError('Face requires a direction')
        if reply['action'] in ('attack', 'transfer', 'return') and not reply['target']:
            raise ValueError('Action requires a target')
        return cls(request.request_id, 'ai', request.request_id, request.frame_id,
                   request.map_epoch, request.scene_id, request.issued_at, request.valid_until,
                   reply['action'], reply['target'], x, reply['direction'], duration, reply['reason'])


@dataclass(frozen=True)
class ActionResult:
    action_id: str
    status: str
    reason: str
    t: float
    frame_id: int | None
    evidence: str = 'observed_state'

    def data(self):
        return asdict(self)


class ActionExecutor:
    """Execute a chosen target without running the rule policy's target selector."""
    def __init__(self, base, allow_transfers=False, allow_combat_transit=False,allow_map_transit=False):
        self.base = base
        self.allow_transfers = allow_transfers
        self.transit_platforms=set(base.combat_platforms or ()) if allow_combat_transit else set()
        self.allow_map_transit=allow_map_transit
        self.current = None
        self.started = None
        self.applied_at = None
        self.origin = None
        self.last_progress = None
        self.progress_at = None
        self.arrived_since = None
        self.results = []
        self.last_submitted = None

    @property
    def busy(self):
        rope = getattr(self.base, 'rope_climber', None)
        return bool(self.base.transition or rope and rope.phase not in ('select', 'approach', 'brake', 'failed'))

    def clearance(self, o):
        speed=self.base.motion.speed if self.base.motion.calibrated else 600
        return max(self.base.motion.edge_margin,o.position_quantum,speed*.1+15)

    def turn_clearance(self,o):
        speed=self.base.motion.speed if self.base.motion.calibrated else 600
        return max(self.base.motion.edge_margin,o.position_quantum,speed*.06+o.position_quantum*.5+5)

    def result(self, status, reason, now, o=None, intent=None, evidence='observed_state'):
        owner = intent or self.current
        if owner:
            self.results.append(ActionResult(owner.action_id, status, reason, now,
                                              o.frame_id if o else None, evidence).data())

    def finish(self, status, reason, now, o=None):
        owned = self.current is not None
        self.result(status, reason, now, o)
        self.current = None
        self.started = self.applied_at = self.origin = self.last_progress = self.progress_at = None
        self.arrived_since = None
        if owned:
            self.base.transition = self.base.pending_edge = self.base.rope_climber = self.base.rope_edge = None
            self.base.pending_since = None
            self.base.jump_brake_edge = self.base.drop_brake_edge = None

    def cancel(self, now, reason, o=None):
        self.finish('cancelled', reason, now, o)

    def attack_members(self, o, direction):
        """Current visible targets inside a direction explicitly chosen by AI.

        This is a group action, never an alias for an expired monster identity.
        No cached sprite or automatic change of direction can sustain it.
        """
        floor=standing_platform(o)
        self.base.active_combat_platforms=[q for q in o.platforms
            if self.base.combat_platforms is not None and q.id in self.base.combat_platforms]
        lane=self.base.firing_lanes.get(floor.id) if floor else None
        self.base.active_firing_lane=((lane['direction'],next((q for q in o.platforms
            if q.id==lane['target']),None)) if lane else None)
        if (not floor or not o.player or abs(o.player.vy)>=60
                or self.base.safe_platforms is not None and floor.id not in self.base.safe_platforms
                or self.base.firing_platforms is not None and floor.id not in self.base.firing_platforms):return []
        p=o.player.box
        return [m for m in o.monsters if (m.box.cx>p.cx)==(direction=='right')
                and self.base.clear_ranged_target(m,p,o.monsters)]

    def rejection(self, intent, o, now, offset, scene_id):
        if intent.action == 'farm':
            return 'intent_hierarchical_mode_required'
        if (intent.source != 'ai' or intent.action not in ACTIONS
                or intent.map_epoch != o.map_epoch or intent.scene_id != scene_id):
            return 'intent_scene_changed'
        if intent.frame_id > o.frame_id or not intent.issued_at <= now < intent.valid_until:
            return 'intent_expired_or_future'
        if o.reason or not o.player or not o.motion_valid or not 0 <= now-o.captured_at < .085:
            return 'intent_observation_invalid'
        if self.busy:
            return 'executor_committed_transfer'
        if (self.current and self.current.action in ('attack','fire') and self.applied_at is not None
                and now-self.applied_at<self.base.attack_hold_seconds):
            return 'executor_committed_attack'
        floor = standing_platform(o)
        if intent.action=='fire':
            if not floor or abs(o.player.vy)>=60:return 'intent_not_grounded'
            if self.base.safe_platforms is not None and floor.id not in self.base.safe_platforms:
                return 'intent_outside_task_platforms'
            if self.base.firing_platforms is not None and floor.id not in self.base.firing_platforms:
                return 'intent_outside_firing_platforms'
            if any(self.base.blocking_close(m,o.player.box) and
                   (abs(m.box.cx-o.player.box.cx)<8 or (m.box.cx>o.player.box.cx)==(intent.direction=='right'))
                   for m in o.monsters):return 'intent_fire_blocked_close'
        if intent.action == 'attack':
            group=intent.target in ('lane:left','lane:right')
            if intent.target.startswith('lane:') and not group:return 'intent_unknown_attack_lane'
            if group and intent.direction not in ('',intent.target.split(':')[1]):
                return 'intent_attack_lane_direction_mismatch'
            target = next((m for m in o.monsters if str(m.track_id) == intent.target), None)
            if not group and (target is None or target.confidence < .78):
                return 'intent_target_lost'
            if not floor or abs(o.player.vy) >= 60:
                return 'intent_not_grounded'
            if self.base.safe_platforms is not None and floor.id not in self.base.safe_platforms:
                return 'intent_outside_task_platforms'
            if self.base.firing_platforms is not None and floor.id not in self.base.firing_platforms:
                return 'intent_outside_firing_platforms'
            if group and not self.attack_members(o,intent.target.split(':')[1]):
                return 'intent_attack_lane_empty'
        if intent.action in ('move', 'face'):
            if not floor or abs(o.player.vy) >= 60:
                return 'intent_not_grounded'
            if intent.action == 'move':
                x = intent.world_x + offset[0]
                margin = self.clearance(o)
                if not floor.left+margin <= x <= floor.right-margin:
                    return 'intent_move_outside_support'
        if intent.action in ('transfer', 'return'):
            if not self.allow_transfers:
                return 'intent_transfers_disabled'
            if floor is None or not self.base.motion.calibrated:
                return 'intent_transfer_uncalibrated'
            platforms = self.transfer_platforms(o)
            if intent.target not in {p.id for p in platforms}:
                return 'intent_transfer_outside_task'
            if floor.id != intent.target and not route(graph(platforms, o.ropes, self.base.motion),
                                                       floor.id, intent.target, self.base.failures):
                return 'intent_transfer_unreachable'
        return None

    def accept(self, intent, o, now, offset=(0, 0), scene_id=''):
        reason = self.rejection(intent, o, now, offset, scene_id)
        if reason:
            self.result('rejected', reason, now, o, intent)
            return False
        if self.current:
            self.finish('cancelled', 'superseded_by_policy', now, o)
        self.current = intent
        self.started = now
        self.origin = (o.player.box.cx-offset[0], o.player.box.y2-offset[1])
        self.last_progress = self.origin
        self.progress_at = now
        self.result('accepted', 'current_observation_validated', now, o)
        return True

    def transfer_platforms(self, o):
        if self.allow_map_transit:self.transit_platforms={p.id for p in o.platforms}
        if self.base.safe_platforms is None:
            return o.platforms
        floor = standing_platform(o)
        return [p for p in o.platforms if (p.id in self.base.safe_platforms or p.id in self.transit_platforms
                    or floor and p.id == floor.id)
                and (floor and p.id==floor.id or p.id in self.transit_platforms or not any(m.confidence >= .78
                    and p.left-20 <= m.box.cx <= p.right+20 and abs(m.box.y2-p.y) < 40
                    for m in [*o.monsters, *o.navigation_targets]))]

    def acknowledge(self, decision, applied, now, o):
        self.last_submitted = dict(t=now, keys=sorted(decision.keys), applied=bool(applied))
        began = applied and self.current and self.applied_at is None
        if began and (self.current.action not in ('attack','fire') or 'shift' in decision.keys):
            self.applied_at = now
            self.result('started', 'input_submission_accepted', now, o, evidence='input_submission')

    def step(self, o, now, offset=(0, 0)):
        if not self.current:
            return Decision(reason='policy_waiting')
        i = self.current
        if o.reason or not o.player or not o.motion_valid or not 0 <= now-o.captured_at < .085:
            self.cancel(now, o.reason or 'invalid_observation', o)
            return Decision(reason='policy_invalid_observation')
        if o.map_epoch != i.map_epoch:
            self.cancel(now, 'map_changed', o)
            return Decision(reason='policy_map_changed')
        p, floor = o.player.box, standing_platform(o)
        elapsed = now-self.started
        if elapsed > (30 if i.action in ('transfer', 'return') else i.duration+1):
            self.finish('failed', 'execution_timeout', now, o)
            return Decision(reason='policy_execution_timeout')
        if i.action == 'wait':
            if elapsed >= i.duration:
                self.finish('completed', 'wait_elapsed', now, o)
            return Decision(reason='policy_wait')
        if i.action == 'move':
            x = i.world_x+offset[0]
            if floor is None or not floor.left <= x <= floor.right or abs(o.player.vy) >= 60:
                self.finish('failed', 'support_changed', now, o)
                return Decision(reason='policy_move_support_changed')
            tolerance = max(8, o.position_quantum)
            dx = x-p.cx
            if abs(dx) <= tolerance:
                if abs(o.player.vx) > 60:
                    return Decision(reason='policy_move_brake')
                if self.arrived_since is None:
                    self.arrived_since = now
                if now-self.arrived_since >= .15:
                    self.finish('completed', 'position_observed', now, o)
                return Decision(reason='policy_move_settle')
            self.arrived_since = None
            side = 'right' if dx > 0 else 'left'
            room = floor.right-p.cx if side == 'right' else p.cx-floor.left
            margin = self.clearance(o)
            if room <= margin:
                self.finish('failed', 'support_clearance_lost', now, o)
                return Decision(reason='policy_move_edge')
            if abs(dx) <= abs(o.player.vx)*.15+tolerance:
                return Decision(reason='policy_move_brake')
            return Decision(frozenset({side}), 'policy_move', i.target)
        if i.action == 'face':
            if not floor or abs(o.player.vy) >= 60:
                self.finish('failed', 'support_changed', now, o)
                return Decision(reason='policy_face_support_changed')
            room = floor.right-p.cx if i.direction == 'right' else p.cx-floor.left
            if room <= self.turn_clearance(o):
                self.finish('failed', 'turn_has_no_clearance', now, o)
                return Decision(reason='policy_face_edge')
            if self.applied_at is not None and now-self.applied_at >= .06:
                self.finish('completed', 'direction_submitted_visual_facing_unknown', now, o)
                return Decision(reason='policy_face_submitted')
            return Decision(frozenset({i.direction}), 'policy_face')
        if i.action=='fire':
            if (floor is None or abs(o.player.vy)>=60
                    or self.base.safe_platforms is not None and floor.id not in self.base.safe_platforms
                    or self.base.firing_platforms is not None and floor.id not in self.base.firing_platforms):
                self.finish('failed','fire_support_changed',now,o)
                return Decision(reason='policy_fire_invalid_support')
            if any(self.base.blocking_close(m,p) and
                   (abs(m.box.cx-p.cx)<8 or (m.box.cx>p.cx)==(i.direction=='right')) for m in o.monsters):
                self.finish('failed','fire_blocked_by_visible_close_monster',now,o)
                return Decision(reason='policy_fire_blocked_close')
            if abs(o.player.vx)>60:return Decision(reason='policy_fire_brake')
            if self.applied_at is not None and now-self.applied_at>=max(i.duration,self.base.attack_hold_seconds):
                self.finish('completed','fire_interval_submitted_hit_unknown',now,o)
                return Decision(reason='policy_fire_submitted')
            if i.direction!=self.base.applied_facing or now-self.base.applied_face_at<.06:
                room=floor.right-p.cx if i.direction=='right' else p.cx-floor.left
                if room<=self.turn_clearance(o):
                    self.finish('failed','turn_has_no_clearance',now,o)
                    return Decision(reason='policy_fire_turn_edge')
                return Decision(frozenset({i.direction}),'policy_fire_face')
            return Decision(frozenset({'shift'}),'policy_fire')
        if i.action == 'attack':
            if i.target in ('lane:left','lane:right'):
                side=i.target.split(':')[1]
                if not self.attack_members(o,side):
                    self.finish('failed','chosen_attack_lane_empty',now,o)
                    return Decision(reason='policy_attack_lane_empty')
                if abs(o.player.vx)>60:return Decision(reason='policy_attack_brake',target=i.target)
                if self.applied_at is not None and now-self.applied_at>=max(i.duration,self.base.attack_hold_seconds):
                    self.finish('completed','attack_interval_submitted_hit_unknown',now,o)
                    return Decision(reason='policy_attack_submitted')
                if side!=self.base.applied_facing or now-self.base.applied_face_at<.06:
                    room=floor.right-p.cx if side=='right' else p.cx-floor.left
                    if room<=self.turn_clearance(o):
                        self.finish('failed','turn_has_no_clearance',now,o)
                        return Decision(reason='policy_attack_turn_edge')
                    return Decision(frozenset({side}),'policy_attack_lane_face',i.target)
                return Decision(frozenset({'shift'}),'policy_attack_lane',i.target)
            target = next((m for m in o.monsters if str(m.track_id) == i.target and m.confidence >= .78), None)
            if target is None:
                self.finish('failed', 'target_lost_not_a_kill', now, o)
                return Decision(reason='policy_attack_target_lost')
            # Geometry and task masks are preconditions, not a new target choice.
            self.base.active_combat_platforms = [q for q in o.platforms
                if self.base.combat_platforms is not None and q.id in self.base.combat_platforms]
            lane = self.base.firing_lanes.get(floor.id) if floor else None
            self.base.active_firing_lane = ((lane['direction'], next((q for q in o.platforms
                if q.id == lane['target']), None)) if lane else None)
            if (floor is None or abs(o.player.vy) >= 60
                    or self.base.safe_platforms is not None and floor.id not in self.base.safe_platforms
                    or self.base.firing_platforms is not None and floor.id not in self.base.firing_platforms
                    or not self.base.clear_ranged_target(target, p, o.monsters)):
                self.finish('failed', 'attack_preconditions_changed', now, o)
                return Decision(reason='policy_attack_invalid_geometry')
            if abs(o.player.vx) > 60:
                return Decision(reason='policy_attack_brake', target=i.target)
            if self.applied_at is not None and now-self.applied_at >= max(i.duration,self.base.attack_hold_seconds):
                self.finish('completed', 'attack_interval_submitted_hit_unknown', now, o)
                return Decision(reason='policy_attack_submitted')
            d = self.base.orient_attack(o, Decision(frozenset({'shift'}), 'policy_attack', i.target), now)
            if d.keys.intersection({'left', 'right'}):
                side = 'right' if 'right' in d.keys else 'left'
                room = floor.right-p.cx if side == 'right' else p.cx-floor.left
                if room <= self.turn_clearance(o):
                    self.finish('failed', 'turn_has_no_clearance', now, o)
                    return Decision(reason='policy_attack_turn_edge')
            return d
        return self.transfer(o, now)

    def transfer(self, o, now):
        """Reuse navigation primitives, but never call Controller.decide."""
        b, i = self.base, self.current
        floor = standing_platform(o)
        if b.rope_climber:
            d = b.rope_climber.decide(o, now)
            if d.reason == 'climb_complete':
                if b.rope_edge:
                    b.verified_edges.add(b.rope_edge[:2])
                b.rope_climber = b.rope_edge = b.transition = None
            elif b.rope_climber.phase == 'failed':
                if b.rope_edge:
                    b.fail_edge(b.rope_edge[:2], now, 'rope')
                self.finish('failed', d.reason, now, o)
            return d
        if b.transition:
            edge = b.transition
            if floor and abs(o.player.vy) < 60:
                if floor.id == edge[1]:
                    if b.landing_floor != floor.id:
                        b.landing_floor = floor.id
                        b.landing_since = now
                    if b.landing_since is None:
                        b.landing_since = now
                    if now-b.landing_since < (.18 if o.position_quantum else .05):
                        return Decision(reason='confirm_landing', target=floor.id)
                    b.verified_edges.add(edge[:2])
                    b.transition = None
                    b.landing_since = None
                elif floor.id != edge[0]:
                    b.fail_edge(edge[:2], now, edge[2])
                    self.finish('failed', 'landed_elsewhere', now, o)
                    return Decision(reason='policy_transfer_wrong_floor')
            if b.transition:
                if now-b.transition_started > 2.5:
                    b.fail_edge(edge[:2], now, edge[2])
                    self.finish('failed', 'transition_timeout', now, o)
                    return Decision(reason='policy_transfer_timeout')
                return b._navigate_step(o, edge, now, airborne=True,transit_platforms=self.transit_platforms)
        if floor is None:
            return Decision(reason='policy_transfer_wait_floor')
        if floor.id == i.target and abs(o.player.vy) < 60:
            if self.arrived_since is None:
                self.arrived_since = now
            if now-self.arrived_since >= .18:
                self.finish('completed', 'destination_landing_observed', now, o)
            return Decision(reason='policy_transfer_confirm', target=floor.id)
        self.arrived_since = None
        b.expire_failures(now)
        path = route(graph(self.transfer_platforms(o), o.ropes, b.motion), floor.id, i.target, b.failures)
        if not path:
            self.finish('failed', 'route_no_longer_valid', now, o)
            return Decision(reason='policy_transfer_unreachable')
        if b.pending_edge != path[0]:
            b.pending_edge, b.pending_since = path[0], now
        if b.pending_since is not None and now-b.pending_since > 4:
            b.fail_edge(path[0][:2], now, path[0][2])
            self.finish('failed', 'launch_approach_timeout', now, o)
            return Decision(reason='policy_transfer_approach_timeout')
        return b._navigate_step(o, path[0], now,transit_platforms=self.transit_platforms)
