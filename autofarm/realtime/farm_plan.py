"""AI selects an entire farming goal; local primitives execute it continuously."""
from collections import deque
from dataclasses import replace

from .actions import ActionExecutor
from .control import standing_platform
from .control import graph, route
from .model import Decision


def suspended_rope_origin(o):
    """A unique mapped rope head is a possible origin, not confirmed attachment."""
    if (standing_platform(o) or not o.player or abs(o.player.vx)>20 or abs(o.player.vy)>20):
        return None
    candidates=[]
    for index,r in enumerate(o.ropes):
        if abs(r.x-o.player.box.cx)>max(12,o.position_quantum):continue
        if not r.top+12<o.player.box.y2<=r.bottom+12:continue
        heads=[p for p in o.platforms if p.left+20<r.x<p.right-20 and abs(p.y-r.top)<15]
        if len(heads)==1:candidates.append((index,heads[0]))
    return candidates[0] if len(candidates)==1 else None


class FarmPlanExecutor(ActionExecutor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.motor = ActionExecutor(*args, **kwargs)
        self.phase = None
        self.attack_seconds = 0.
        self.last_ack = None
        self.review_sent = False
        self.outcomes = deque(maxlen=12)
        self.reward = None
        self.net_exp = 0
        self.reward_unknown = False
        self.phase_started = None
        self.resume = None

    @property
    def busy(self):
        if self.current and self.current.action == 'farm':
            return self.phase != 'fire' or self.motor.busy
        return super().busy

    def review_due(self):
        if not self.current:
            return True
        if self.current.action != 'farm':
            return False
        return (self.phase == 'fire' and not self.review_sent
                and self.attack_seconds >= min(25., self.current.duration / 2))

    def rejection(self, intent, o, now, offset, scene_id):
        if intent.action != 'farm':
            if intent.action != 'wait':
                return 'intent_choose_farm_or_wait'
            return super().rejection(intent, o, now, offset, scene_id)
        # Validate provenance, calibrated route and grounded origin with the same motor contract.
        floor = standing_platform(o)
        origin = suspended_rope_origin(o) if not floor else None
        check = replace(intent, action='wait' if origin or floor and floor.id == intent.target else 'transfer',
                        target=intent.target, world_x=None, direction='', duration=3)
        reason = super().rejection(check, o, now, offset, scene_id)
        if reason:
            return reason
        if origin:
            if not self.allow_transfers or not self.base.motion.calibrated:
                return 'intent_rope_resume_disabled_or_uncalibrated'
            ps=self.transfer_platforms(o)
            if origin[1].id!=intent.target and not route(graph(ps,o.ropes,self.base.motion),
                                                       origin[1].id,intent.target,self.base.failures):
                return 'intent_rope_resume_destination_unreachable'
        dest = next((p for p in o.platforms if p.id == intent.target), None)
        if (dest is None or self.base.safe_platforms is not None and dest.id not in self.base.safe_platforms
                or self.base.firing_platforms is not None and dest.id not in self.base.firing_platforms):
            return 'intent_farm_destination_not_firing_support'
        x = intent.world_x + offset[0]
        if not dest.left + self.clearance(o) <= x <= dest.right - self.clearance(o):
            return 'intent_farm_stand_outside_support'
        return None

    def accept(self, intent, o, now, offset=(0, 0), scene_id=''):
        accepted = super().accept(intent, o, now, offset, scene_id)
        if accepted and intent.action == 'farm':
            self.phase = 'travel'
            self.phase_started = now
            self.attack_seconds = 0.
            self.last_ack = None
            self.review_sent = False
            self.net_exp = 0
            self.reward_unknown = self.reward is None
            origin=suspended_rope_origin(o)
            if origin:
                from .climbing import RopeClimber
                self.resume=RopeClimber(target_id=origin[1].id,jump_height=self.base.motion.jump_height,
                    speed=self.base.motion.speed,jump_distance=self.base.motion.jump_distance)
                self.resume.rope_index=origin[0];self.resume.target_id=origin[1].id
                self.resume.started=self.resume.phase_at=now
                self.resume.phase='probe';self.resume.attempts=1
                self.phase='rope_resume'
        return accepted

    def observe_feedback(self, status):
        row = status.get('confirmed_experience')
        if self.current and self.current.action == 'farm':
            if row is None or self.reward is None:
                self.reward_unknown = True
            elif row['t'] != self.reward['t']:
                if (row['level'] != self.reward['level'] or row['exp'] < self.reward['exp']
                        or not 0 < row['t']-self.reward['t'] <= 2):
                    self.reward_unknown = True
                else:
                    self.net_exp += row['exp']-self.reward['exp']
        self.reward = row

    def finish(self, status, reason, now, o=None):
        if self.current and self.current.action == 'farm':
            outcome = dict(action_id=self.current.action_id, platform=self.current.target,
                world_x=self.current.world_x, direction=self.current.direction,
                status=status, reason=reason, elapsed_seconds=now-self.started,
                attack_input_seconds=self.attack_seconds,
                net_exp=None if self.reward_unknown else self.net_exp,
                reward_evidence='calibrated interval only; individual hits/kills unknown', t=now)
            self.outcomes.append(outcome)
            self.results.append(dict(action_id=self.current.action_id, status='plan_outcome',
                reason=reason, t=now, frame_id=o.frame_id if o else None, plan=outcome))
            self.motor.cancel(now, reason, o)
        super().finish(status, reason, now, o)
        self.phase = None
        self.phase_started = None
        self.last_ack = None
        self.resume = None

    def acknowledge(self, decision, applied, now, o):
        if not self.current or self.current.action != 'farm':
            return super().acknowledge(decision, applied, now, o)
        if self.last_ack:
            t, firing = self.last_ack
            if firing and 0 <= now-t <= .1:
                self.attack_seconds += now-t
        self.last_ack = (now, bool(applied and 'shift' in decision.keys))
        self.motor.acknowledge(decision, applied, now, o)
        if applied and self.applied_at is None:
            self.applied_at = now
            self.result('started', 'plan_input_submission_accepted', now, o, evidence='input_submission')

    def state(self):
        return dict(phase=self.phase, attack_input_seconds=self.attack_seconds,
                    net_exp=None if self.reward_unknown else self.net_exp,
                    review_requested=self.review_sent, previous_plans=list(self.outcomes))

    def step(self, o, now, offset=(0, 0)):
        if not self.current or self.current.action != 'farm':
            return super().step(o, now, offset)
        i = self.current
        if (o.reason or not o.player or not o.motion_valid or not 0 <= now-o.captured_at < .085
                or o.map_epoch != i.map_epoch):
            self.cancel(now, 'plan_observation_changed', o)
            return Decision(reason='policy_plan_invalid_observation')
        if now-self.started > i.duration+50 or self.attack_seconds >= i.duration:
            self.finish('completed' if self.attack_seconds >= i.duration else 'failed',
                        'plan_attack_budget_complete' if self.attack_seconds >= i.duration else 'plan_timeout', now, o)
            return Decision(reason='policy_plan_finished')
        if self.phase=='rope_resume':
            d=self.resume.decide(o,now)
            if self.resume.phase=='failed':
                self.finish('failed',d.reason,now,o)
                return Decision(reason='policy_plan_rope_resume_failed')
            if d.reason!='climb_complete':return d
            self.resume=None;self.phase='travel';self.phase_started=now
        floor = standing_platform(o)
        if self.phase in ('position', 'fire') and (not floor or floor.id != i.target):
            self.finish('failed', 'chosen_plan_support_changed', now, o)
            return Decision(reason='policy_plan_support_changed')
        # At most three immediate phase changes; arrival never requires another network reply.
        for _ in range(3):
            if self.motor.current is None:
                floor = standing_platform(o)
                if self.phase == 'travel' and floor and floor.id == i.target:
                    self.phase = 'position'
                    self.phase_started = now
                action = {'travel':'transfer', 'position':'move', 'fire':'fire'}[self.phase]
                child = replace(i, action_id=i.action_id+':'+self.phase, action=action,
                    target=i.target if action=='transfer' else '',
                    world_x=i.world_x if action=='move' else None,
                    direction=i.direction if action=='fire' else '', duration=20 if action=='fire' else 3,
                    issued_at=now, valid_until=now+31)
                if not self.motor.accept(child, o, now, offset, i.scene_id):
                    reason = self.motor.results[-1]['reason']
                    self.motor.results.clear()
                    self.finish('failed', reason, now, o)
                    return Decision(reason='policy_plan_precondition_failed')
            # Positioning can span a wide platform. Keep the primitive's short lease renewed,
            # with a separate bounded phase deadline and all per-frame support checks intact.
            if self.phase == 'position' and now-self.phase_started < 15:
                self.motor.started = now
            d = self.motor.step(o, now, offset)
            failure = next((r for r in self.motor.results if r['status'] in ('failed','rejected')), None)
            done = self.motor.current is None
            self.motor.results.clear()
            if failure:
                self.finish('failed', failure['reason'], now, o)
                return Decision(reason='policy_plan_motor_failed')
            if not done:
                return d
            self.phase = {'travel':'position', 'position':'fire', 'fire':'fire'}[self.phase]
            self.phase_started = now
        return Decision(reason='policy_plan_phase_changed')
