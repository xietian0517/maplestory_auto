"""Shared live/replay policy, execution, acknowledgements and feedback pipeline."""
from collections import Counter,deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path

import cv2

from .actions import ActionExecutor, ActionIntent
from .farm_plan import FarmPlanExecutor
from .model import Box, Decision, MotionProfile
from .control import Controller
from .combat import HasteRefresh
from .recovery import ActiveRecovery
from .policy import AsyncPolicy, create_action_policy, PolicyConfig, RulePolicy
from .feedback import FeedbackTracker, OnlineHUD, StateHistory, observation_data
from .policy_api import PolicyAPIError


CONFIG_FIELDS = ('direct_attacks', 'prefer_jump_attacks', 'jump_attacks_enabled', 'attack_hold_seconds',
    'standing_burst_seconds', 'standing_continue_occluded', 'safe_fire_min', 'sustain_farming',
    'refresh_attack_facing', 'prefer_firing_anchor', 'farm_anchor', 'anchor_quiet_seconds',
    'firing_viewport', 'safe_platforms', 'firing_platforms', 'combat_platforms', 'firing_lanes',
    'policy_mode', 'policy_min_vote', 'policy_allow_stay', 'policy_search_seconds')


def controller_configuration(controller, navigate=False, climb=False):
    base = getattr(controller, 'base', controller)
    values = {}
    for name in CONFIG_FIELDS:
        if hasattr(base, name):
            v = getattr(base, name)
            values[name] = sorted(v) if isinstance(v, set) else v
    return dict(kind=type(controller).__name__,navigate=navigate, climb=climb, motion=asdict(base.motion) if hasattr(base, 'motion') else None,
                values=values, preferred=list(getattr(controller, 'preferred', [])),
                hud_boxes=[asdict(b) for b in getattr(base, 'hud_boxes', [])],
                platform_policy=getattr(getattr(base, 'platform_policy', None), 'artifact', None))


def restore_controller(data):
    motion = MotionProfile.parse(data['motion']) if data.get('motion') else None
    kind=data.get('kind','RopeClimber' if data.get('climb') else 'NavigationController' if data.get('navigate') else 'Controller')
    if kind not in ('RopeClimber','NavigationController','Controller'):
        raise ValueError('Unknown recorded controller kind')
    if kind=='RopeClimber':
        from .climbing import RopeClimber
        controller = RopeClimber()
    elif kind=='NavigationController':
        from .calibration import NavigationController
        controller = NavigationController(motion)
    else:
        controller = Controller(motion, data.get('navigate',False))
    base = getattr(controller, 'base', controller)
    for name, value in data.get('values', {}).items():
        if name not in CONFIG_FIELDS:
            raise ValueError('Unknown recorded controller configuration')
        if name in ('safe_platforms', 'firing_platforms', 'combat_platforms') and value is not None:
            value = set(value)
        setattr(base, name, value)
    controller.preferred = data.get('preferred', [])
    base.hud_boxes = [Box(**b) for b in data.get('hud_boxes', [])]
    if data.get('platform_policy'):
        from .demo_policy import PlatformIntent
        base.platform_policy = PlatformIntent(data['platform_policy'])
    return controller


class ArchivedPolicy:
    """Save exact model inputs off the realtime thread."""
    def __init__(self, policy, folder):
        self.policy, self.folder = policy, Path(folder)

    def propose(self, request):
        dest = self.folder/request.request_id
        dest.mkdir(parents=True, exist_ok=False)
        data = request.data()
        data['images'] = []
        for i, im in enumerate(request.images):
            name = f'{i}.png'
            if not cv2.imwrite(str(dest/name), im):
                raise OSError('Could not archive policy image')
            data['images'].append(name)
        (dest/'request.json').write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
        try:
            intent = self.policy.propose(request)
        except PolicyAPIError as error:
            (dest/'error.json').write_text(json.dumps(error.data(),ensure_ascii=False),encoding='utf-8')
            raise
        (dest/'response.json').write_text(json.dumps(intent.data(), ensure_ascii=False), encoding='utf-8')
        return intent


class DecisionPipeline:
    def __init__(self, controller, config=None, navigate=False, climb=False, folder=None,
                 policy=None, replay=False, recovery=None, haste=None):
        self.config = config or PolicyConfig()
        self.rule = RulePolicy(controller, recovery or ActiveRecovery(), haste or HasteRefresh(), navigate, climb)
        executor_type = FarmPlanExecutor if self.config.hierarchical else ActionExecutor
        self.executor = executor_type(self.rule.base, self.config.allow_transfers,
            self.config.allow_combat_transit,self.config.allow_map_transit)
        self.history = StateHistory()
        self.feedback = FeedbackTracker()
        self.scene_id = None
        self.epoch = None
        self.previous_input = None
        self.counts = Counter()
        self.last_model_latency_ms=None
        self.runner = None
        self.replay = replay
        self.pending = None
        self.delivery = None
        self.cycle_events = []
        self.last_trace = None
        self.hud = self.hud_future = self.hud_request = None
        self.hud_next_at = 0
        self.feedback_generation = 0
        self.hud_pool = None
        self.last_snapshot = None
        self.last_model_error = None
        self.last_model_error_details = None
        self.action_results=deque(maxlen=20)
        self.closed = False
        self.journal = self.event_log = None
        provider = None
        if self.config.mode != 'rule' and not replay:
            provider = policy or create_action_policy(self.config)
        if folder is not None:
            folder = Path(folder)
            if (folder/'feedback_config.json').exists() and not replay:
                self.hud = OnlineHUD(folder)
                self.feedback.max_gap = self.hud.max_gap
            if provider is not None:
                provider = ArchivedPolicy(provider, folder/'policy_requests')
        self.initial_configuration = controller_configuration(controller, navigate, climb)
        try:
            if folder is not None:
                self.journal = (folder/'policy_trace.jsonl').open('w', encoding='utf-8', buffering=1)
                self.event_log = (folder/'feedback_events.jsonl').open('w', encoding='utf-8', buffering=1)
                self.journal.write(json.dumps(dict(kind='configuration', version=1,
                    controller=self.initial_configuration, policy=self.config.data()))+'\n')
            if self.hud:
                self.hud_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='hud-feedback')
            if provider is not None:
                self.runner = AsyncPolicy(provider, self.config)
        except Exception:
            self.close()
            raise

    @property
    def base(self):
        return self.rule.base

    def emit(self, event):
        if event.get('event')=='policy_response':self.last_model_latency_ms=event.get('latency_ms')
        if event.get('event')=='policy_error':
            self.last_model_error=event['error_type']
            self.last_model_error_details=event.get('error')
            self.counts['model_errors']+=1
        elif event.get('event')=='policy_response':
            self.last_model_error=None
            self.last_model_error_details=None
        elif event.get('event')=='policy_response_rejected':
            self.counts['expired_responses']+=1
        elif event.get('event')=='action_result':
            self.counts['action_'+event['status']]+=1
            self.action_results.append(dict(event))
        self.cycle_events.append(event)
        if self.event_log:
            self.event_log.write(json.dumps(event, ensure_ascii=False)+'\n')

    def suspend(self, now, reason):
        self.feedback_generation += 1
        if self.runner:
            self.runner.invalidate(now, reason)
        self.executor.cancel(now, reason)
        self.history.reset()
        self.action_results.clear()
        self.feedback.invalidate()
        self.previous_input = None

    def reset(self, now, reason):
        self.suspend(now, reason)
        self.rule.reset()
        if self.journal:
            self.journal.write(json.dumps(dict(kind='reset', t=now, reason=reason))+'\n')

    def set_scene(self, scene_id, now):
        if scene_id == self.scene_id:
            return
        self.suspend(now, 'scene_changed')
        self.scene_id = scene_id

    def releases(self, releases):
        jump = getattr(self.base, 'jump_combat', None)
        for key, released_at, token in releases:
            if key == 'alt' and jump and jump.started == token:
                jump.jump_released_at = released_at

    def step(self, o, now, image=None, offset=(0, 0), scene_id='', health=None,
             override=None, override_source=None, releases=(), delivery=None, readings=None, width=1366):
        self.cycle_events = []
        self.delivery = delivery
        self.executor.results = []
        changed = self.scene_id != scene_id
        self.set_scene(scene_id, now)
        self.releases(releases)
        if o is not None and self.epoch != o.map_epoch:
            if self.epoch is not None:
                self.suspend(now, 'map_epoch_changed')
            self.epoch = o.map_epoch
        valid = bool(o and o.player and not o.reason and o.motion_valid and 0 <= now-o.captured_at < .085)
        if not valid:
            self.suspend(now, o.reason if o and o.reason else 'invalid_observation')
        for event in self.feedback.observe(o if valid else None, now, offset, self.previous_input, health):
            self.emit(event)
        hud_delivery = readings
        if self.hud_future is not None and self.hud_future.done():
            sampled_at, requested_scene, requested_epoch, requested_generation = self.hud_request
            try:
                reading = self.hud_future.result()
            except Exception:
                reading = dict(exp=None, level=None, reason='hud_reader_error')
            if (valid and requested_scene == scene_id and requested_epoch == o.map_epoch
                    and requested_generation == self.feedback_generation
                    and 0 <= now-sampled_at <= self.feedback.max_gap):
                hud_delivery = dict(reading=reading, sampled_at=sampled_at)
            self.hud_future = self.hud_request = None
        if hud_delivery and valid:
            for event in self.feedback.hud(hud_delivery['reading'], hud_delivery['sampled_at'], now, o):
                self.emit(event)
        if self.hud and valid and self.hud.scene_id == scene_id and image is not None and self.hud_future is None and now >= self.hud_next_at:
            self.hud_request = (o.captured_at, scene_id, o.map_epoch, self.feedback_generation)
            self.hud_future = self.hud_pool.submit(self.hud.read, image.copy())
            self.hud_next_at = now+self.hud.interval
        if valid:
            self.history.observe(o, now, offset, self.previous_input,
                                 image if self.config.mode != 'rule' else None)
            if self.config.hierarchical:
                self.executor.observe_feedback(self.feedback.status(now))
        proposals = []
        if override is not None:
            self.executor.cancel(now, override_source or 'external_override', o)
            if self.runner:
                self.runner.invalidate(now, override_source or 'external_override')
            decision, source = override, override_source or 'override'
        elif self.config.mode == 'rule':
            decision, proposals = self.rule.decide(o, now, image.shape[1] if image is not None else width, offset)
            source = proposals[-1]['source']
        elif not valid:
            decision, source = Decision(reason=o.reason if o else 'waiting_for_gpt'), 'observation_guard'
        else:
            intent = delivery
            if self.runner:
                intent = self.runner.poll(now)
                for event in self.runner.events:
                    self.emit(event)
                self.runner.events.clear()
            self.delivery = intent
            if intent:
                self.counts['responses'] += 1
                if self.config.mode == 'active':
                    accepted = self.executor.accept(intent, o, now, offset, scene_id)
                    self.counts['accepted' if accepted else 'rejected'] += 1
                else:
                    rejection = self.executor.rejection(intent, o, now, offset, scene_id)
                    self.emit(dict(event='shadow_advice', t=now, intent=intent.data(), eligible=rejection is None,
                                   rejection=rejection))
            if self.config.mode == 'shadow':
                decision, proposals = self.rule.decide(o, now, image.shape[1] if image is not None else width, offset)
                source = proposals[-1]['source']
            elif self.executor.current is not None:
                decision, source = self.executor.step(o, now, offset), 'ai_executor'
            elif self.config.fallback == 'rule':
                decision, proposals = self.rule.decide(o, now, image.shape[1] if image is not None else width, offset)
                source = 'fallback_rule'
            else:
                decision, source = Decision(reason='policy_waiting'), 'fallback_wait'
            self.last_snapshot = self.history.snapshot(o, now, offset, self.base, self.executor,
                self.feedback.status(now), self.config.allow_transfers)
            self.last_snapshot['action_results']=[e for e in self.action_results if 0<=now-e['t']<=20]
            self.last_snapshot['model_latency_ms']=self.last_model_latency_ms
            self.last_snapshot['hierarchical']=self.config.hierarchical
            if self.config.hierarchical:
                self.last_snapshot['available_actions']=['farm','wait']
                self.last_snapshot['plan_progress']=self.executor.state()
            request_due = (not self.config.hierarchical or self.config.mode == 'shadow'
                           or self.executor.review_due())
            if self.runner and not self.executor.busy and request_due:
                images,stamps=self.history.policy_images(o,now,image)
                self.last_snapshot['image_observed_at']=stamps
                if self.runner.submit(o, now, scene_id, self.last_snapshot, images):
                    self.counts['requests'] += 1
                    if self.config.hierarchical and self.executor.current:
                        self.executor.review_sent = True
                for event in self.runner.events:
                    self.emit(event)
                self.runner.events.clear()
        if not valid and (override is None or decision.keys):
            if o and not 0 <= now-o.captured_at < .085:
                decision = Decision(reason='stale_frame')
            elif self.config.mode!='rule' or override is not None:
                decision = Decision(reason=o.reason if o and o.reason else 'waiting_for_gpt' if o is None else 'invalid_observation')
            source = 'observation_guard'
        if self.runner:
            for event in self.runner.events:
                self.emit(event)
            self.runner.events.clear()
        # Active AI actions cannot be silently replaced by idle recovery or Buff.
        if (valid and self.config.mode == 'active' and self.config.refresh_buff
                and not self.executor.busy and override is None):
            changed_buff = self.rule.haste.apply(o, decision, now, self.rule.controller)
            if changed_buff != decision:
                proposals.append(dict(source='buff', reason=changed_buff.reason, keys=sorted(changed_buff.keys),
                                      target=changed_buff.target))
                self.emit(dict(event='action_overridden', t=now, by='buff', original_source=source))
                decision, source = changed_buff, 'buff'
        for result in self.executor.results:
            self.emit(dict(event='action_result', **result))
        self.executor.results.clear()
        self.counts['cycles'] += 1
        self.counts['source_'+source] += 1
        self.pending = dict(kind='cycle', t=now, scene_id=scene_id, observation=observation_data(o),
            width=image.shape[1] if image is not None else width,
            offset=list(offset), health=health, hud_delivery=hud_delivery,
            releases=list(releases), delivered_intent=self.delivery.data() if self.delivery else None,
            override=None if override is None else dict(keys=sorted(override.keys), reason=override.reason, target=override.target),
            override_source=override_source, decision=dict(keys=sorted(decision.keys), reason=decision.reason, target=decision.target),
            proposed_decision=dict(keys=sorted(decision.keys), reason=decision.reason, target=decision.target),
            source=source, proposals=proposals, events=list(self.cycle_events),
            action=self.executor.current.data() if self.executor.current else None,
            configuration=controller_configuration(self.rule.controller, self.rule.navigate, self.rule.climb)
                if changed else None)
        return decision

    def commit(self, decision, applied, now, o=None, held_keys=(), simulated=False, owner=None,
               external_owner=False):
        actual=dict(keys=sorted(decision.keys), reason=decision.reason, target=decision.target)
        if self.pending and actual!=self.pending['decision']:
            original_source=self.pending['source']
            self.pending['decision']=actual
            self.pending['source']='frame_guard'
            self.counts['source_'+original_source]-=1
            self.counts['source_frame_guard']+=1
            self.emit(dict(event='action_overridden',t=now,by='frame_guard',original_source=original_source))
            self.suspend(now,decision.reason or 'frame_guard')
        acknowledged = bool(applied or simulated)
        if acknowledged:
            if owner is not None:
                if hasattr(owner, 'acknowledge'):
                    owner.acknowledge(decision, now)
            elif not external_owner:
                self.rule.acknowledge(decision, now)
                if self.pending and self.pending['source'] == 'ai_executor':
                    self.executor.acknowledge(decision, True, now, o)
        for result in self.executor.results:
            if simulated:
                result = dict(result, evidence='simulated_input_submission')
            self.emit(dict(event='action_result', **result))
        self.executor.results.clear()
        self.previous_input = dict(t=now, applied=bool(applied), simulated=bool(simulated),
                                  requested_keys=sorted(decision.keys), held_keys=sorted(held_keys))
        if self.pending:
            self.pending.update(input=self.previous_input, acknowledged=acknowledged,
                                input_owner='external' if owner is not None or external_owner else 'rule',
                                events=list(self.cycle_events), action=self.executor.current.data() if self.executor.current else None)
            self.last_trace = self.pending
            if self.journal:
                self.journal.write(json.dumps(self.pending, ensure_ascii=False)+'\n')
            self.pending = None

    def status(self):
        return dict(mode=self.config.mode, fallback=self.config.fallback, counts=dict(self.counts),
                    hierarchical=self.config.hierarchical,
                    plan_progress=self.executor.state() if self.config.hierarchical else None,
                    pending_model=bool(self.runner and self.runner.future),
                    last_model_error=getattr(self,'last_model_error',None),
                    last_model_error_details=self.last_model_error_details,
                    model_blocked=bool(self.runner and self.runner.blocked_error),
                    action=self.executor.current.data() if self.executor.current else None,
                    last_source=self.last_trace['source'] if self.last_trace else None)

    def close(self, now=0):
        if self.closed:
            return
        self.closed = True
        self.cycle_events = []
        self.executor.cancel(now, 'session_end')
        for result in self.executor.results:
            self.emit(dict(event='action_result', **result))
        self.executor.results.clear()
        if self.runner:
            self.runner.close()
        if self.hud_pool:
            self.hud_pool.shutdown(wait=False, cancel_futures=True)
        if self.journal:
            self.journal.write(json.dumps(dict(kind='session_end',t=now,events=self.cycle_events),
                                          ensure_ascii=False)+'\n')
            self.journal.close()
        if self.event_log:
            self.event_log.close()
