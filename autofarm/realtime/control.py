"""Pure navigation/combat decisions and independent, expiring input leases."""
from collections import deque
import threading
import time

from .model import Decision, MotionProfile


def standing_platform(o):
    if not o.player: return None
    if abs(o.player.vy)>140: return None
    x, y = o.player.box.cx, o.player.box.y2
    possible = [p for p in o.platforms if p.left <= x <= p.right and abs(p.y-y) < 16]
    return min(possible, key=lambda p: abs(p.y-y)) if possible else None


def graph(platforms, ropes, motion):
    """Candidate edges; success is recorded separately, never inferred as proven."""
    result = {p.id: [] for p in platforms}
    for a in platforms:
        for b in platforms:
            if a.id == b.id: continue
            overlap = min(a.right, b.right) - max(a.left, b.left)
            gap = max(0, b.left-a.right, a.left-b.right)
            dy = b.y-a.y
            if abs(dy) < 8 and gap < 8:
                result[a.id].append((b.id, 'walk'))
            elif 18 < dy <= 240 and overlap > 32:
                result[a.id].append((b.id, 'drop'))
            elif motion.calibrated and -motion.jump_height*.8 <= dy <= 180 and gap+40 <= motion.jump_distance+max(0,dy)*.5:
                result[a.id].append((b.id, 'jump'))
            elif motion.calibrated and dy < -30 and any(
                    a.left+12 < r.x < a.right-12 and b.left+12 < r.x < b.right-12
                    and abs(r.top-min(a.y,b.y)) < 20
                    and -15 <= max(a.y,b.y)-r.bottom <= max(0,motion.jump_height-12) for r in ropes):
                result[a.id].append((b.id, 'rope'))
    return result


def route(edges, start, goal, blocked=()):
    queue = deque([(start, [])]); seen = {start}
    while queue:
        node, path = queue.popleft()
        if node == goal: return path
        for nxt, kind in edges.get(node, []):
            if nxt not in seen and (node,nxt) not in blocked:
                seen.add(nxt); queue.append((nxt, path+[(node,nxt,kind)]))
    return []


class Controller:
    def __init__(self, motion=None, navigate=False):
        self.motion = motion or MotionProfile()
        self.navigate = navigate
        self.facing = None
        self.turn_until = 0
        self.applied_facing=None; self.applied_face_at=0
        self.transition = None
        self.transition_started = 0
        self.failures = set()
        self.failure_retry_at={}; self.failure_counts={}
        self.verified_edges = set()
        self.last_epoch = None
        self.no_enemy_since = None
        self.last_progress = None
        self.preferred = []
        self.patrol_side='left'
        self.rope_climber=None
        self.pending_edge=None
        self.rope_edge=None
        self.pending_since=None
        self.landing_since=None
        self.invalid_since=None
        self.rope_recovery=None
        self.last_grounded_at=None
        self.explore_until=0; self.explore_origin=None; self.explored={}
        self.attack_burst_at=None; self.reposition_until=0; self.reposition_keys=frozenset()
        self.jump_combat=None; self.next_jump_combat=0; self.prefer_jump_attacks=False

    def reset(self):
        self.facing = None; self.turn_until = 0; self.transition = None
        self.applied_facing=None; self.applied_face_at=0
        self.no_enemy_since = None; self.last_progress = None
        self.rope_climber=None
        self.pending_edge=None
        self.rope_edge=None
        self.pending_since=None
        self.landing_since=None
        self.invalid_since=None
        self.rope_recovery=None
        self.last_grounded_at=None
        self.attack_burst_at=None; self.reposition_until=0
        self.jump_combat=None

    def request_exploration(self,o,now):
        floor=standing_platform(o)
        if not floor or abs(o.player.vy)>=60: return False
        self.jump_combat=None
        self.pending_edge=None; self.pending_since=None; self.transition=None; self.rope_climber=None
        self.rope_edge=None; self.last_progress=None
        self.no_enemy_since=now-2; self.explore_until=now+10; self.explore_origin=floor.id
        self.explored[floor.id]=self.explored.get(floor.id,0)+1
        return True

    def orient_attack(self,o,d,now):
        """Gate every attack on a current target and an accepted direction input."""
        takeoff=d.reason=='jump_attack_takeoff' and self.jump_combat is not None
        if 'shift' not in d.keys and not takeoff: return d
        if not o or not o.player or o.reason or not o.motion_valid or now-o.captured_at>=.085:
            return Decision(reason='attack_wait_fresh_target')
        target_id=str(self.jump_combat.target_id) if takeoff else d.target
        target=next((m for m in o.monsters if str(m.track_id)==target_id and m.confidence>=.78),None)
        if target is None: return Decision(reason='attack_target_lost')
        dx=target.box.cx-o.player.box.cx
        if abs(dx)<8: return Decision(reason='attack_target_crossing')
        face='right' if dx>0 else 'left'
        if face!=self.applied_facing or now-self.applied_face_at<.06:
            return Decision(frozenset({face}),'attack_face_confirm',target_id)
        return d

    def acknowledge(self,d,now):
        face=next((k for k in ('left','right') if k in d.keys),None)
        if face is not None:
            if face!=self.applied_facing: self.applied_face_at=now
            self.applied_facing=face; self.facing=face
        if self.jump_combat: self.jump_combat.on_input_applied(d,now)

    def fight(self,o,floor,now):
        p=o.player.box
        targets=[m for m in o.monsters if m.confidence>=.78 and abs(m.box.y2-p.y2)<35
                 and abs(m.box.cx-p.cx)<self.motion.attack_max]
        if not targets:
            self.attack_burst_at=None; self.reposition_until=0
            return None
        self.no_enemy_since=None
        target=min(targets,key=lambda m:abs(m.box.cx-p.cx))
        dx=target.box.cx+max(-40,min(40,target.vx*.1))-p.cx
        face='right' if dx>=0 else 'left'
        if now<self.reposition_until:
            self.facing=next(iter(self.reposition_keys),self.facing)
            return Decision(self.reposition_keys,'attack_reposition')
        if self.attack_burst_at is not None and now-self.attack_burst_at>3:
            self.attack_burst_at=None
            if floor:
                choices=[(p.cx-floor.left,'left'),(floor.right-p.cx,'right')]
                room,side=max(choices)
                if room>55:
                    self.reposition_keys=frozenset({side}); self.reposition_until=now+.22
                    self.facing=side
                    return Decision(self.reposition_keys,'attack_reposition')
        if abs(dx)<self.motion.attack_min and floor:
            away='left' if dx>=0 else 'right'
            room=p.cx-floor.left if away=='left' else floor.right-p.cx
            margin=max(self.motion.edge_margin,self.motion.speed*.1+15)
            if room>margin+20:
                self.facing=away
                return Decision(frozenset({away}),'retreat')
        if self.facing!=face:
            self.facing=face; self.turn_until=now+.04
        if now<self.turn_until: return Decision(frozenset({face}),'turn' if floor else 'turn_without_floor_map')
        if self.attack_burst_at is None: self.attack_burst_at=now
        return Decision(frozenset({'shift'}),'attack' if floor else 'attack_without_floor_map',str(target.track_id))

    def fail_edge(self,edge,now,kind=None):
        self.failures.add(edge)
        self.failure_counts[edge]=self.failure_counts.get(edge,0)+1
        self.failure_retry_at[edge]=now+(float("inf") if kind=="jump" and self.failure_counts[edge]>=2 else min(4,self.failure_counts[edge]))

    def decide(self, o, now):
        if o.map_epoch != self.last_epoch:
            self.reset(); self.failures.clear(); self.verified_edges.clear(); self.last_epoch = o.map_epoch
            self.failure_retry_at.clear(); self.failure_counts.clear()
            self.explore_until=0; self.explore_origin=None; self.explored.clear()
        for edge,retry in list(self.failure_retry_at.items()):
            if now>=retry:
                self.failures.discard(edge); del self.failure_retry_at[edge]
        if o.reason or not o.player or not o.motion_valid:
            grounded_at=self.last_grounded_at
            if self.invalid_since is None: self.invalid_since=now
            if not o.motion_valid or now-self.invalid_since>.4:
                searching_since=self.no_enemy_since
                self.reset()
                # An interrupted observation must not restart the idle delay
                # each time the sprite becomes visible. Actions still require
                # a fresh identified player and a valid terrain observation.
                if o.motion_valid:
                    self.no_enemy_since=searching_since; self.last_grounded_at=grounded_at
            return Decision(reason=o.reason or 'player_or_camera_unknown')
        if now-o.captured_at >= .085:
            return Decision(reason='stale_frame')
        self.invalid_since=None
        p = o.player.box
        floor = standing_platform(o)
        if floor and abs(o.player.vy)<60: self.last_grounded_at=now
        if (self.prefer_jump_attacks and floor and abs(o.player.vy)<60 and self.rope_climber
                and self.rope_climber.phase in ('select','approach','brake')
                and any(m.confidence>=.78 and abs(m.box.y2-p.y2)<35
                        and abs(m.box.cx-p.cx)<self.motion.attack_max for m in o.monsters)):
            self.rope_climber=None; self.rope_edge=None
        if self.jump_combat:
            d=self.jump_combat.decide(o,now); self.facing=self.jump_combat.facing
            if self.jump_combat.done:
                self.jump_combat=None; self.next_jump_combat=now+.25
            return d
        if (floor and abs(o.player.vy)<60 and not self.transition and not self.rope_climber
                and now>=self.next_jump_combat):
            higher_limit=self.motion.jump_height+20 if self.motion.calibrated else 35
            jump_targets=[m for m in o.monsters if m.confidence>=.78 and abs(m.box.cx-p.cx)<self.motion.attack_max
                          and (35<=p.y2-m.box.y2<=higher_limit or
                               abs(m.box.y2-p.y2)<35 and (self.prefer_jump_attacks or abs(m.box.cx-p.cx)<self.motion.attack_min))]
            if jump_targets:
                from .combat import JumpAttack
                target=min(jump_targets,key=lambda m:abs(m.box.cx-p.cx))
                self.pending_edge=None; self.pending_since=None; self.no_enemy_since=None
                self.jump_combat=JumpAttack(floor,target,now,self.applied_facing if self.prefer_jump_attacks else self.facing,self.motion.attack_max)
                d=self.jump_combat.decide(o,now);self.facing=self.jump_combat.facing
                return d
        # Combat outranks route bookkeeping. A ranged target does not need to
        # share our platform; finish an airborne/attached movement first.
        if abs(o.player.vy)<60 and (floor or not self.transition and not self.rope_climber):
            if self.prefer_jump_attacks and floor and now<self.next_jump_combat:
                return Decision(reason="jump_attack_cooldown")
            fight=self.fight(o,floor,now)
            if fight: return fight
        if self.rope_climber:
            d=self.rope_climber.decide(o,now)
            if d.reason=='climb_complete':
                if self.rope_edge: self.verified_edges.add(self.rope_edge[:2])
                self.rope_climber=None; self.transition=None
                self.rope_edge=None
            elif self.rope_climber.phase=='failed':
                if self.rope_edge: self.fail_edge(self.rope_edge[:2],now)
                self.rope_climber=None; self.rope_edge=None; self.transition=None
            return d
        if self.transition:
            a,b,kind = self.transition
            if floor and floor.id not in (a,b) and abs(o.player.vy)<60:
                self.fail_edge((a,b),now,kind); self.transition=None; self.landing_since=None
                return Decision(reason='landed_elsewhere_replan')
            if floor and floor.id == b and abs(o.player.vy)<60:
                if self.landing_since is None: self.landing_since=now
                if now-self.landing_since<.05: return Decision(reason='confirm_landing',target=b)
                self.verified_edges.add((a,b)); self.transition = None; self.landing_since=None
                self.facing=None
            elif now-self.transition_started > 2.5:
                self.fail_edge((a,b),now,kind); self.transition = None
                self.landing_since=None
                return Decision(reason='route_failed_replan')
            else:
                self.landing_since=None
                return self._navigate_step(o, self.transition, now, airborne=True)
        if self.pending_edge:
            if not floor: return Decision(reason='approach_wait_for_floor')
            # An approach may be interrupted by knockback onto another floor.
            if floor and floor.id!=self.pending_edge[0]:
                self.pending_edge=None; self.pending_since=None
            elif self.pending_since is not None and now-self.pending_since>4:
                self.fail_edge(self.pending_edge[:2],now); self.pending_edge=None; self.pending_since=None
                return Decision(reason='launch_approach_timeout')
            elif not any(m.confidence>=.78 and abs(m.box.y2-p.y2)<35 and
                         65<abs(m.box.cx-p.cx)<self.motion.attack_max for m in o.monsters):
                return self._navigate_step(o,self.pending_edge,now)
            else:
                self.pending_edge=None; self.pending_since=None
        if not floor:
            # Recover a stationary character already hanging on a mapped rope
            # after identity loss or a semantic scene refresh.
            suspended=[(i,r,q) for i,r in enumerate(o.ropes) for q in o.platforms
                       if abs(p.cx-r.x)<12 and r.top+30<p.y2<r.bottom+50
                       and abs(q.y-r.top)<15 and q.left+20<r.x<q.right-20]
            if suspended and abs(o.player.vx)<35 and abs(o.player.vy)<35:
                i,r,q=min(suspended,key=lambda item:abs(item[1].x-p.cx))
                if self.rope_recovery is None or self.rope_recovery[:2]!=(i,q.id):
                    self.rope_recovery=(i,q.id,now)
                elif now-self.rope_recovery[2]>=.15:
                    from .climbing import RopeClimber
                    self.rope_climber=RopeClimber(target_id=q.id,jump_height=self.motion.jump_height,speed=self.motion.speed,jump_distance=self.motion.jump_distance)
                    self.rope_climber.target_id=q.id; self.rope_climber.rope_index=i
                    self.rope_climber.started=now; self.rope_climber.phase_at=now
                    self.rope_climber.phase='ascend'; self.rope_climber.attempts=1
                    return Decision(frozenset({'up'}),'rope_resume',q.id)
            else: self.rope_recovery=None
            return Decision(reason='airborne_or_floor_unknown')
        self.rope_recovery=None
        # Include travel during the maximum accepted frame/lease age. The
        # uncalibrated case must assume haste may be active.
        margin = max(self.motion.edge_margin,(self.motion.speed if self.motion.calibrated else 600)*.1+15)
        margin=min(margin,(floor.right-floor.left)/3)
        if p.cx < floor.left+margin:
            self.facing='right'
            return Decision(frozenset({'right'}), 'edge_recovery')
        if p.cx > floor.right-margin:
            self.facing='left'
            return Decision(frozenset({'left'}), 'edge_recovery')
        if self.no_enemy_since is None: self.no_enemy_since=now
        if not self.navigate or now-self.no_enemy_since < 1.5:
            return Decision(reason='search')
        if self.last_progress is None or abs(p.cx-self.last_progress[0])>8:
            self.last_progress=(p.cx,now)
        elif now-self.last_progress[1]>2:
            self.reset(); return Decision(reason='stuck_request_strategy')
        # Move within the observed platform towards a visible monster first.
        distant=[m for m in o.monsters if abs(m.box.y2-p.y2)<35 and floor.left+margin<m.box.cx<floor.right-margin]
        if distant:
            m=min(distant,key=lambda m:abs(m.box.cx-p.cx))
            direction='right' if m.box.cx>p.cx else 'left'
            self.facing=direction
            return Decision(frozenset({direction}), 'approach')
        edges=graph(o.platforms,o.ropes,self.motion)
        goals = [] if floor.id in self.preferred else list(self.preferred)
        if self.explore_until>now and floor.id==self.explore_origin:
            goals=[q.id for q in sorted(o.platforms,key=lambda q:(self.explored.get(q.id,0),
                    -(q.right-q.left),abs(q.y-floor.y))) if q.id!=floor.id]+goals
        elif floor.id!=self.explore_origin: self.explore_until=0
        enemy_floors=[]
        for m in o.navigation_targets:
            candidates=[q for q in o.platforms if q.id!=floor.id and q.left<=m.box.cx<=q.right
                        and abs(q.y-m.box.y2)<45]
            if candidates:
                q=min(candidates,key=lambda q:abs(q.y-m.box.y2))
                if q.id not in enemy_floors: enemy_floors.append(q.id)
        goals=enemy_floors+goals
        if not self.motion.calibrated or floor.right-floor.left<180:
            # Find room to measure motion when the spawn ledge is too small or
            # occluded; downward overlap is visible without a jump estimate.
            goals += [q.id for q in sorted(o.platforms,key=lambda p:p.right-p.left,reverse=True)
                      if q.id!=floor.id and q.right-q.left>floor.right-floor.left+50]
        for goal in goals:
            path=route(edges,floor.id,goal,self.failures)
            if path:
                self.pending_edge=path[0]
                return self._navigate_step(o,path[0],now)
        if floor.right-floor.left<180 and any(a==floor.id for a,b in self.failure_retry_at):
            return Decision(reason='route_retry_pending')
        # Search only within the presently observed continuous platform. No
        # inferred off-screen extension or uncalibrated jump is required.
        inset=min(margin+20,max(5,(floor.right-floor.left)/2-20))
        target=floor.left+inset if self.patrol_side=='left' else floor.right-inset
        if abs(p.cx-target)<8:
            self.patrol_side='right' if self.patrol_side=='left' else 'left'
            self.last_progress=None
            return Decision(reason='patrol_turnaround')
        self.facing='right' if target>p.cx else 'left'
        return Decision(frozenset({self.facing}),'patrol')

    def _navigate_step(self, o, edge, now, airborne=False):
        a,b,kind=edge
        if self.pending_since is None: self.pending_since=now
        source=next((p for p in o.platforms if p.id==a),None)
        dest=next((p for p in o.platforms if p.id==b),None)
        if not source or not dest:
            self.transition=None; self.pending_edge=None; self.pending_since=None
            return Decision(reason='route_invalid')
        if kind=='rope':
            from .climbing import RopeClimber
            self.rope_climber=RopeClimber(target_id=dest.id,jump_height=self.motion.jump_height,speed=self.motion.speed,jump_distance=self.motion.jump_distance)
            self.rope_edge=edge
            self.pending_edge=None
            return self.rope_climber.decide(o,now)
        x=o.player.box.cx
        if kind=='drop' and self.failure_counts.get((a,b),0):
            exits=[q for q in (source.left-18,source.right+18) if dest.left+24<q<dest.right-24]
            if exits:
                goal=min(exits,key=lambda q:abs(q-x))
                if self.transition is None:
                    self.transition=edge; self.pending_edge=None; self.transition_started=now; self.pending_since=None
                direction='right' if goal>x else 'left'
                self.facing=direction
                return Decision(frozenset({direction}) if abs(goal-x)>8 else frozenset(),'walk_off_drop',b)
        mid=max(dest.left+24,min(dest.right-24,x))
        lo=max(source.left,dest.left)+15; hi=min(source.right,dest.right)-15
        launch=max(source.left+18,min(source.right-18,mid))
        if kind=='drop' and lo<hi: launch=(lo+hi)/2
        if not airborne and abs(x-launch)>8:
            face='right' if launch>x else 'left'; self.facing=face
            return Decision(frozenset({face}), 'approach_launch',b)
        if self.transition is None:
            self.transition=edge; self.pending_edge=None; self.transition_started=now
            self.pending_since=None
        elapsed=now-self.transition_started
        direction='right' if mid>x else 'left'
        if kind=='walk': return Decision(frozenset({direction}), 'walk_route',b)
        if kind=='drop':
            # Establish Down before the jump edge; an unordered simultaneous
            # chord can be interpreted as a normal jump by the game.
            keys={'down'} if elapsed<.40 else set()
            if .12<=elapsed<.26: keys.add('alt')
            return Decision(frozenset(keys),'drop',b)
        keys={direction} if abs(mid-x)>12 else set()
        if elapsed<.080: keys.add('alt')
        return Decision(frozenset(keys), 'jump',b)


class LeasedKeys:
    """No network/model dependency; watchdog releases on timeout or focus change."""
    ALLOWED=frozenset({'left','right','up','down','shift','alt','home'})

    def __init__(self, adapter, hwnd, clock=time.perf_counter):
        self.adapter=adapter; self.hwnd=hwnd; self.clock=clock
        self.held=set(); self.deadline=0; self.epoch=0
        self.lock=threading.RLock(); self.stopped=threading.Event(); self.error=None
        self.thread=None; self.blocked=False
        self.pulses={}; self.release_events=[]; self.suppressed=set()

    def allowed(self):
        return self.adapter.is_target_foreground(self.hwnd) and not self.adapter.emergency_pressed()

    def _release(self,key):
        self.adapter.send_key(key,True); self.held.discard(key)
        pulse=self.pulses.pop(key,None)
        if pulse:
            self.release_events.append((key,self.clock(),pulse[1])); self.suppressed.add(key)

    def pop_releases(self):
        with self.lock:
            events=self.release_events; self.release_events=[]; return events

    def clear(self):
        with self.lock:
            errors=[]
            for key in tuple(self.held):
                try: self._release(key)
                except Exception as e: errors.append(e)
            if errors: raise RuntimeError('Failed to release one or more keys') from errors[0]

    def check(self):
        with self.lock:
            allowed=self.allowed()
            if not allowed and not self.blocked: self.epoch+=1
            self.blocked=not allowed
            if self.adapter.emergency_pressed(): self.stopped.set()
            if not allowed or self.clock()>=self.deadline or self.stopped.is_set(): self.clear()
            for key,(at,token) in list(self.pulses.items()):
                if self.clock()>=at and key in self.held: self._release(key)

    def apply(self, keys, captured_at, epoch, pulses=None):
        with self.lock:
            now=self.clock(); keys=set(keys)
            if not keys<=self.ALLOWED or {'left','right'}<=keys or {'up','down'}<=keys:
                self.clear(); raise ValueError('Invalid key state')
            if self.stopped.is_set() or epoch!=self.epoch or not self.allowed() or not 0<=now-captured_at<.085:
                self.clear(); return False
            self.suppressed.intersection_update(keys)
            if pulses:
                for key,(duration,token) in pulses.items():
                    if key not in keys or not 0<duration<=.08: raise ValueError("Invalid pulse")
                    self.suppressed.discard(key)
            keys-=self.suppressed
            # Frame age plus lease never exceeds 100ms from capture start.
            self.deadline=min(captured_at+.095,now+.060)
            for key in tuple(self.held-keys): self._release(key)
            self.held.intersection_update(keys)
            for key in sorted(keys-self.held,key=lambda k:(k in ('alt','shift','home'),k)):
                self.held.add(key)
                try: self.adapter.send_key(key,False)
                except Exception:
                    self.stopped.set(); self.clear(); raise
            for key,(duration,token) in (pulses or {}).items():
                self.pulses[key]=(self.clock()+duration,token)
            return True

    def _watch(self):
        while not self.stopped.wait(.005):
            try: self.check()
            except Exception as e:
                self.error=type(e).__name__; self.stopped.set()
        try: self.clear()
        except Exception as e: self.error=type(e).__name__

    def __enter__(self):
        self.thread=threading.Thread(target=self._watch,daemon=True); self.thread.start(); return self

    def __exit__(self,*args):
        self.stopped.set(); self.thread.join(timeout=1); self.clear()
