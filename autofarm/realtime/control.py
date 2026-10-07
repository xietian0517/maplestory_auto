"""Pure navigation/combat decisions and independent, expiring input leases."""
from collections import deque
import threading
import time

from .model import Decision, MotionProfile


def standing_platform(o):
    if not o.player: return None
    if abs(o.player.vy)>140: return None
    x, y = o.player.box.cx, o.player.box.y2
    # The name/sprite centre and the foothold endpoint differ by a few pixels.
    # Only tolerate this at low vertical speed, never across a real gap.
    edge_tolerance = max(12,o.position_quantum+4) if abs(o.player.vy) < 60 else 0
    # The calibrated marker may sit up to one cell above the physical feet.
    # A marker well BELOW a platform is different: it is an airborne character
    # passing under that platform, including the low-velocity jump apex.
    below_tolerance=max(4,o.position_quantum*.3) if o.position_quantum else 16
    possible = [p for p in o.platforms if p.left-edge_tolerance <= x <= p.right+edge_tolerance
                and -16<y-p.y<below_tolerance]
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
                    a.left+12 < r.x < a.right-12 and b.left+20 < r.x < b.right-20
                    and abs(r.top-min(a.y,b.y)) < 15
                    and max(a.y,b.y)-r.bottom <= max(0,motion.jump_height-12) for r in ropes):
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
        self.drop_brake_edge=None
        self.drop_brake_stable=None
        self.jump_brake_edge=None
        self.jump_brake_stable=None
        self.landing_since=None
        self.landing_floor=None;self.jump_origin_clearance=0.;self.jump_lift_confirmed=False
        self.invalid_since=None
        self.rope_recovery=None
        self.rope_resume_blocked={}
        self.last_grounded_at=None
        self.explore_until=0; self.explore_origin=None; self.explored={}
        self.attack_burst_at=None; self.reposition_until=0; self.reposition_keys=frozenset()
        self.jump_combat=None; self.next_jump_combat=0; self.prefer_jump_attacks=False
        # 跳A 默认关闭：实测 run_20260927_113138 中 jump_attack_* 占 4646 帧
        # (≈154.9 s / 25.8%)，其中 align_height 1520、target_occluded 1106、
        # brake 608；这段时间既没打死目标，又打断了移动与站位。人工示范全程
        # 不用跳A。只有显式选择 pure-jump 策略时才会打开。
        self.jump_attacks_enabled=False
        self.direct_attacks=False
        self.direct_hold_until=0; self.direct_rest_until=0; self.direct_target=None
        self.hud_boxes=[];self.visibility_exit=None
        self.direct_last_seen=None
        self.standing_commit_at=None;self.standing_commit_face=None;self.standing_last_applied=None
        self.standing_seen=None
        self.standing_last_target=None;self.standing_floor=None
        self.standing_burst_seconds=.5;self.standing_continue_occluded=False
        # 已提交的攻击不能因为下一帧迟到或短暂漏检被切成一瞬间的按键。
        # 该时长是攻击键按住的默认下限，可由 farm_plan/命令行/GUI 覆盖。
        self.attack_hold_seconds=.30
        # 安全射击距离：猴子会主动走过来，等它进 65px 近战判定再让位时，
        # 这一刀已经变成短刀（用户实测的“无效攻击”），人也被顶着打。
        # 取值与选位用的同层净空一致，选出来的点天然在让位距离之外，
        # 不会出现“选中却不敢开火”的死区。
        self.safe_fire_min=85.
        self.prefer_firing_anchor=False
        self.refresh_attack_facing=False;self.attack_face_until=0.;self.attack_face_refresh_at=None
        self.sustain_farming=False;self.firing_viewport=(1366,694);self.firing_goal=None
        self.firing_bounds={};self.firing_diagnostics=None
        self.safe_platforms=None;self.firing_platforms=None
        self.firing_lanes={};self.active_firing_lane=None
        self.combat_platforms=None;self.active_combat_platforms=[]
        self.safe_quiet_since=None;self.safe_quiet_floor=None
        self.platform_policy=None;self.policy_mode='shadow';self.policy_min_vote=.55;self.policy_advice=None
        self.policy_allow_stay=False
        self.farm_anchor=None;self.anchor_quiet_seconds=6.;self.anchor_cooldown_until=0.
        self.policy_search_seconds=0.;self.policy_search_floor=None;self.policy_search_until=0.;self.policy_search_retry=0.

    def reset(self):
        self.active_firing_lane=None;self.active_combat_platforms=[]
        self.firing_bounds={};self.firing_diagnostics=None
        self.firing_goal=None
        self.safe_quiet_since=None;self.safe_quiet_floor=None
        self.attack_face_until=0.;self.attack_face_refresh_at=None
        self.facing = None; self.turn_until = 0; self.transition = None
        self.applied_facing=None; self.applied_face_at=0
        self.no_enemy_since = None; self.last_progress = None
        self.rope_climber=None
        self.pending_edge=None
        self.rope_edge=None
        self.pending_since=None
        self.drop_brake_edge=None
        self.drop_brake_stable=None
        self.jump_brake_edge=None
        self.jump_brake_stable=None
        self.landing_since=None
        self.landing_floor=None;self.jump_origin_clearance=0.;self.jump_lift_confirmed=False
        self.invalid_since=None
        self.rope_recovery=None
        self.last_grounded_at=None
        self.attack_burst_at=None; self.reposition_until=0
        self.jump_combat=None
        self.direct_hold_until=0; self.direct_rest_until=0; self.direct_target=None
        self.direct_last_seen=None
        self.standing_commit_at=None;self.standing_commit_face=None;self.standing_last_applied=None
        self.standing_seen=None
        self.standing_last_target=None;self.standing_floor=None

        self.policy_search_floor=None;self.policy_search_until=0.;self.policy_search_retry=0.

    def request_exploration(self,o,now):
        if self.safe_platforms is not None:return False
        floor=standing_platform(o)
        if not floor or abs(o.player.vy)>=60: return False
        if self.farm_anchor==floor.id:self.anchor_cooldown_until=now+20
        self.policy_search_floor=None;self.policy_search_until=0.
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
        if target is None:
            if d.reason=='direct_attack_occlusion_hold' and self.can_hold_occluded(o,now):return d
            if d.reason=='attack_burst_continue' and self.can_continue_standing(o,now):return d
            return Decision(reason='attack_target_lost')
        dx=target.box.cx-o.player.box.cx
        if abs(dx)<8: return Decision(reason='attack_target_crossing')
        face='right' if dx>0 else 'left'
        if self.refresh_attack_facing:
            floor=standing_platform(o)
            room=(floor.right-o.player.box.cx if face=='right' else o.player.box.cx-floor.left) if floor else 0
            if room>self.motion.speed*.18+self.motion.edge_margin and abs(o.player.vy)<60:
                if (now<self.attack_face_until or self.attack_face_refresh_at is None
                        or now-self.attack_face_refresh_at>=1.4):
                    # The direction model already believes `face`, so this is a
                    # re-assert, not a turn: keep the attack this frame was
                    # already allowed to send and add the direction key to it.
                    # Live run_20260927_113138 spent 45.5 s in 380 re-assert
                    # episodes whose first frame already matched `face`; 0.4.4's
                    # own review warned this could lower the attack duty cycle.
                    # Only grounded standing attacks are merged; a real turn and
                    # every jump/climb frame keep the two-phase gate below.
                    if ('shift' in d.keys and face==self.applied_facing
                            and d.reason in ('attack','attack_without_floor_map','attack_burst_continue')):
                        return Decision(frozenset(set(d.keys)|{face}),'attack_face_confirm',target_id)
                    return Decision(frozenset({face}),'attack_face_confirm',target_id)
        if face!=self.applied_facing or now-self.applied_face_at<.06:
            return Decision(frozenset({face}),'attack_face_confirm',target_id)
        return d

    def acknowledge(self,d,now):
        face=next((k for k in ('left','right') if k in d.keys),None)
        if (self.refresh_attack_facing and d.reason=='attack_face_confirm' and face is not None
                and (now>=self.attack_face_until or face!=self.applied_facing)):
            self.attack_face_refresh_at=now;self.attack_face_until=now+.18
        if face is not None:
            if face!=self.applied_facing: self.applied_face_at=now
            self.applied_facing=face; self.facing=face
        if self.jump_combat: self.jump_combat.on_input_applied(d,now)
        if d.reason in ('attack','attack_without_floor_map','attack_burst_continue') and 'shift' in d.keys:
            if (self.standing_last_applied is None or now-self.standing_last_applied>.18
                    or self.standing_commit_face!=self.applied_facing
                    or self.standing_commit_at is None or now-self.standing_commit_at>=self.standing_burst_seconds):
                self.standing_commit_at=now;self.standing_commit_face=self.applied_facing
            self.standing_last_applied=now
        if d.reason=='direct_attack_start':
            # One continuous hold, long enough for roughly two skill animations.
            # Count from accepted input, not a proposed/blocked attack.
            self.direct_hold_until=now+1.4
            self.direct_rest_until=self.direct_hold_until+.15
            self.direct_target=d.target

    def standing_target(self,m,p):
        if self.combat_platforms is not None:
            from .firing_position import on_combat_platform
            if not on_combat_platform(m,self.active_combat_platforms):return False
        if self.firing_lanes:
            from .firing_position import in_firing_lane
            if self.active_firing_lane is None:return False
            direction,target=self.active_firing_lane
            if not in_firing_lane(m,p.cx,direction,target):return False
        # Human demo 20.4s and 23.7s: standing on the leaf, throws hit
        # monsters ~60-70px below. Keep the unverified upward limit narrow.
        return (m.confidence>=.78 and -35<m.box.y2-p.y2<85
                and abs(m.box.cx-p.cx)<self.motion.attack_max)

    def melee_close(self,m,p):
        return abs(m.box.cx-p.cx)<self.motion.attack_min and abs(m.box.y2-p.y2)<35

    def blocking_close(self,m,p):
        """近到会顶掉飞镖的距离。比 65px 近战判定宽松，留出后撤时间。"""
        return (m.confidence>=.78
                and abs(m.box.cx-p.cx)<max(self.motion.attack_min,self.safe_fire_min)
                and abs(m.box.y2-p.y2)<35)

    def clear_ranged_target(self,m,p,monsters):
        if not self.standing_target(m,p) or self.blocking_close(m,p):return False
        dx=m.box.cx-p.cx
        # Choosing a distant target cannot bypass a close enemy in the same
        # firing direction: the game still resolves the key as melee.
        return not any(self.blocking_close(n,p)
                       and (abs(n.box.cx-p.cx)<8 or (n.box.cx-p.cx)*dx>0) for n in monsters)

    def fight(self,o,floor,now):
        if self.safe_platforms is not None and (floor is None or floor.id not in self.safe_platforms):
            return None
        if self.firing_platforms is not None and (floor is None or floor.id not in self.firing_platforms):return None
        if self.firing_lanes and (floor is None or floor.id not in self.firing_lanes):return None
        p=o.player.box
        targets=[m for m in o.monsters if self.standing_target(m,p)]
        if not targets:
            if floor and self.can_continue_standing(o,now):
                return Decision(frozenset({'shift'}),'attack_burst_continue',self.standing_seen[4])
            if not self.direct_attacks and floor and self.wait_standing_target(o,now):
                # A 40–90ms detection gap previously inserted walking between
                # attacks (round 068 frames 154–174). Pause navigation briefly;
                # only a freshly detected target can issue another attack.
                return Decision(reason='attack_reacquire_wait')
            if self.sustain_farming and floor and self.wait_engagement(o,now):
                return Decision(reason='engagement_reacquire_wait')
            self.attack_burst_at=None; self.reposition_until=0
            return None
        if o.position_quantum and abs(o.player.vx)>60:
            return Decision(reason='attack_brake')
        self.no_enemy_since=None
        clear=[m for m in targets if self.clear_ranged_target(m,p,o.monsters)]
        # Preserve the accepted firing direction while targets remain clear.
        # The monkey preset extends this window: the reviewed leaf-platform
        # demonstration holds Shift for 1.79 and 2.55 seconds.
        if (self.standing_commit_at is not None and now-self.standing_commit_at<self.standing_burst_seconds
                and self.standing_last_applied is not None and now-self.standing_last_applied<=.18
                and self.standing_commit_face==self.applied_facing):
            same_side=[m for m in clear if (m.box.cx>p.cx)==(self.standing_commit_face=='right')]
            if same_side:clear=same_side
        target=min(clear or targets,key=lambda m:abs(m.box.cx-p.cx))
        dx=target.box.cx+max(-40,min(40,target.vx*.1))-p.cx
        face='right' if dx>=0 else 'left'
        if self.direct_attacks:
            # Close contact turns the ranged attack into weak melee and repeated
            # knockback. Human demonstrations first make room on the same floor.
            if floor and self.melee_close(target,p):
                away='left' if dx>=0 else 'right'
                room=p.cx-floor.left if away=='left' else floor.right-p.cx
                margin=max(self.motion.edge_margin,self.motion.speed*.1+15)
                if room>margin+25:
                    return Decision(frozenset({away}),'ranged_make_room',str(target.track_id))
            self.remember_direct_target(o,target,now)
            return Decision(frozenset({'shift'}),'direct_attack_start',str(target.track_id))
        # 周围有怪就发不出飞镖（用户实测）：怪进到安全射击距离就不再按
        # Shift，先沿台面让位；背后还有空间时才让位，否则维持原判定。
        if floor and self.blocking_close(target,p):
            away='left' if dx>=0 else 'right'
            room=p.cx-floor.left if away=='left' else floor.right-p.cx
            margin=max(self.motion.edge_margin,self.motion.speed*.1+15)
            if room>margin+20:
                self.facing=away
                return Decision(frozenset({away}),'retreat',str(target.track_id))
        if now<self.reposition_until:
            self.facing=next(iter(self.reposition_keys),self.facing)
            return Decision(self.reposition_keys,'attack_reposition')
        if (self.attack_burst_at is not None and now-self.attack_burst_at>3
                and not (self.sustain_farming and clear)):
            self.attack_burst_at=None
            if floor:
                choices=[(p.cx-floor.left,'left'),(floor.right-p.cx,'right')]
                room,side=max(choices)
                if room>55:
                    self.reposition_keys=frozenset({side}); self.reposition_until=now+.22
                    self.facing=side
                    return Decision(self.reposition_keys,'attack_reposition')
        if self.facing!=face:
            self.facing=face; self.turn_until=now+.04
        if now<self.turn_until: return Decision(frozenset({face}),'turn' if floor else 'turn_without_floor_map')
        if self.attack_burst_at is None: self.attack_burst_at=now
        self.standing_seen=(now,p.cx,p.y2,face,str(target.track_id))
        self.standing_last_target=target;self.standing_floor=floor.id if floor else None
        return Decision(frozenset({'shift'}),'attack' if floor else 'attack_without_floor_map',str(target.track_id))

    def can_continue_standing(self,o,now):
        # Only bridge a short visual gap in an already submitted burst.
        # The last visible timestamp is never renewed by an inferred hold.
        if (not self.standing_continue_occluded or self.direct_attacks
                or self.standing_commit_at is None
                or not 0<=now-self.standing_commit_at<self.standing_burst_seconds
                or not self.wait_standing_target(o,now) or not self.standing_last_target):return False
        floor=standing_platform(o)
        if not floor or floor.id!=self.standing_floor:return False
        p=o.player.box;target=self.standing_last_target
        return (self.clear_ranged_target(target,p,o.monsters)
                and not any(m.confidence>=.78 and (m.track_id==target.track_id or self.melee_close(m,p))
                            for m in o.monsters))

    def wait_engagement(self,o,now):
        if (not self.standing_seen or self.standing_last_applied is None
                or not o.player or o.reason or not o.motion_valid):return False
        at,x,y,face,target=self.standing_seen;p=o.player.box
        floor=standing_platform(o)
        return (0<=now-at<=.65 and 0<=now-self.standing_last_applied<=.85
                and floor is not None and floor.id==self.standing_floor
                and abs(p.cx-x)<=12 and abs(p.y2-y)<=8 and abs(o.player.vy)<60)

    def firing_anchor_available(self,o,edges,floor):
        """Use only the reviewed firing ledge, with current target evidence."""
        anchor=next((q for q in o.platforms if q.id==self.farm_anchor),None)
        if anchor is None:return False
        monsters=[m for m in [*o.monsters,*o.navigation_targets] if m.confidence>=.78]
        if any(anchor.left-20<=m.box.cx<=anchor.right+20 and abs(m.box.y2-anchor.y)<35 for m in monsters):return False
        center=(anchor.left+anchor.right)/2
        visible_target=any(-35<m.box.y2-anchor.y<85
                           and self.motion.attack_min+20<abs(m.box.cx-center)<self.motion.attack_max-20
                           for m in monsters)
        if not visible_target:return False
        if floor.id==anchor.id:return True
        path=route(edges,floor.id,anchor.id,self.failures)
        return bool(path) and len(path)<=2

    def wait_standing_target(self,o,now):
        if (not self.standing_seen or self.standing_last_applied is None
                or not o.player or o.reason or not o.motion_valid):return False
        at,x,y,face,target=self.standing_seen;p=o.player.box
        grace=.45 if self.safe_platforms is not None else .18
        return (0<=now-at<=grace and at<=self.standing_last_applied<=now
                and self.applied_facing==face and abs(o.player.vy)<60
                and abs(p.cx-x)<=12 and abs(p.y2-y)<=8)

    def remember_direct_target(self,o,target,now):
        p=o.player.box
        self.direct_last_seen=(now,target,p.cx,p.y2,'right' if target.box.cx>p.cx else 'left')

    def can_hold_occluded(self,o,now):
        """Bounded continuation of an acknowledged burst, never a new blind shot."""
        if not self.direct_last_seen or now>=self.direct_hold_until or not o.player:return False
        at,target,x,y,face=self.direct_last_seen;p=o.player.box
        if not self.standing_target(target,p) or self.melee_close(target,p):return False
        if not 0<=now-at<=.18 or self.applied_facing!=face:return False
        if abs(p.cx-x)>12 or abs(p.y2-y)>8 or abs(o.player.vy)>=60:return False
        if any(m.track_id==target.track_id or self.melee_close(m,p)
               for m in o.monsters if m.confidence>=.78):return False
        return True

    def avoid_hud(self,o,floor):
        """Leave reviewed screen-fixed overlays while identity is still visible.

        Only a continuous observed foothold permits this move. No extrapolated
        player, blind escape, gap crossing, or interruption of an active climb.
        """
        if not floor or abs(o.player.vy)>=60 or self.rope_climber or self.transition:
            self.visibility_exit=None;return None
        p=o.player.box;margin=max(28,self.motion.speed*.1+15)
        def reachable(x):return floor.left+margin<=x<=floor.right-margin
        if self.visibility_exit:
            ident,x=self.visibility_exit
            if ident!=floor.id or not reachable(x) or abs(p.cx-x)<12:self.visibility_exit=None
        if self.visibility_exit is None:
            for b in self.hud_boxes:
                if p.y2+20<b.y1 or p.y2-65>b.y2:continue
                if p.cx+55<b.x1 or p.cx-55>b.x2:continue
                exits=[x for x in (b.x1-90,b.x2+90) if reachable(x)]
                if exits:
                    self.visibility_exit=(floor.id,min(exits,key=lambda x:abs(x-p.cx)));break
        if self.visibility_exit:
            self.jump_combat=None;self.direct_hold_until=0;self.direct_rest_until=0;self.direct_target=None
            return Decision(frozenset({'left' if self.visibility_exit[1]<p.cx else 'right'}),'avoid_hud_occlusion')
        return None

    def fail_edge(self,edge,now,kind=None):
        self.failures.add(edge)
        self.failure_counts[edge]=self.failure_counts.get(edge,0)+1
        self.failure_retry_at[edge]=now+(float("inf") if kind in ("jump","rope") and self.failure_counts[edge]>=2 else min(4,self.failure_counts[edge]))

    def expire_failures(self,now):
        for edge,retry in list(self.failure_retry_at.items()):
            if now>=retry:
                self.failures.discard(edge);del self.failure_retry_at[edge]

    def navigation_stalled(self,x,now):
        if self.last_progress is None or abs(x-self.last_progress[0])>8:
            self.last_progress=(x,now)
        elif now-self.last_progress[1]>2:
            self.reset();return True
        return False

    def resume_suspended_rope(self,o,now):
        """Probe a stationary mapped rope pose before attempting combat."""
        p=o.player.box
        suspended=[(i,r,q) for i,r in enumerate(o.ropes) for q in o.platforms
                   if abs(p.cx-r.x)<12 and r.top+12<p.y2<r.bottom+15
                   and abs(q.y-r.top)<15 and q.left+20<r.x<q.right-20
                   and abs((p.y2-r.top)-self.rope_resume_blocked.get((i,q.id),float('inf')))>25]
        if not suspended or abs(o.player.vx)>=35 or abs(o.player.vy)>=35:
            self.rope_recovery=None
            return None
        i,r,q=min(suspended,key=lambda item:abs(item[1].x-p.cx))
        if self.rope_recovery is None or self.rope_recovery[:2]!=(i,q.id):
            self.rope_recovery=(i,q.id,now)
        if now-self.rope_recovery[2]<.15:
            return Decision(reason='rope_observe_attachment',target=q.id)
        from .climbing import RopeClimber
        self.rope_climber=RopeClimber(target_id=q.id,jump_height=self.motion.jump_height,
                                     speed=self.motion.speed,jump_distance=self.motion.jump_distance)
        self.rope_climber.target_id=q.id; self.rope_climber.rope_index=i
        self.rope_climber.started=now; self.rope_climber.phase_at=now
        self.rope_climber.phase='probe'; self.rope_climber.attempts=1
        self.rope_climber.probe_relative=p.y2-r.top
        return Decision(frozenset({'up'}),'rope_probe_attachment',q.id)

    def decide(self, o, now):
        self.policy_advice=None
        if o.map_epoch != self.last_epoch:
            self.reset(); self.failures.clear(); self.verified_edges.clear(); self.last_epoch = o.map_epoch
            self.failure_retry_at.clear(); self.failure_counts.clear()
            self.rope_resume_blocked.clear()
            self.explore_until=0; self.explore_origin=None; self.explored.clear()
        self.expire_failures(now)
        if o.reason or not o.player or not o.motion_valid:
            self.policy_search_floor=None;self.policy_search_until=0.
            grounded_at=self.last_grounded_at
            if self.invalid_since is None: self.invalid_since=now
            if not o.motion_valid or now-self.invalid_since>.4:
                searching_since=self.no_enemy_since
                # Losing the character must not give the same failed transfer
                # an unlimited fresh retry budget when perception recovers.
                if self.rope_climber and self.rope_edge:
                    self.fail_edge(self.rope_edge[:2],now,'rope')
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
        self.active_combat_platforms=[q for q in o.platforms
            if self.combat_platforms is not None and q.id in self.combat_platforms]
        self.active_firing_lane=None
        if floor and floor.id in self.firing_lanes:
            lane=self.firing_lanes[floor.id]
            target=next((q for q in o.platforms if q.id==lane['target']),None)
            if target:self.active_firing_lane=(lane['direction'],target)
        if floor is None or floor.id!=self.policy_search_floor:
            self.policy_search_floor=None;self.policy_search_until=0.
        visibility=None if self.safe_platforms is not None else self.avoid_hud(o,floor)
        if visibility:return visibility
        if (self.sustain_farming and self.navigate and floor and abs(o.player.vy)<60
                and not self.transition and not self.rope_climber and not self.jump_combat
                and not (self.safe_platforms is not None and self.pending_edge)):
            engaged=self.wait_engagement(o,now)
            visible=any(self.clear_ranged_target(m,p,o.monsters) for m in o.monsters)
            quiet_wait=None
            if self.safe_platforms is not None and floor.id in self.firing_platforms:
                if self.safe_quiet_floor!=floor.id or visible or engaged:
                    self.safe_quiet_since=now;self.safe_quiet_floor=floor.id
                if not visible and not engaged and now-self.safe_quiet_since<self.anchor_quiet_seconds:
                    quiet_wait=Decision(reason='safe_perch_wait_respawn',target=floor.id)
            if engaged and not visible and not any(self.blocking_close(m,p) for m in o.monsters):
                if self.can_continue_standing(o,now):
                    return Decision(frozenset({'shift'}),'attack_burst_continue',self.standing_seen[4])
                return Decision(reason='engagement_reacquire_wait')
            if not engaged:
                from .firing_position import choose_position
                self.firing_diagnostics={}
                choice=choose_position(o,floor,self.motion,self.failures,self.firing_viewport,self.firing_diagnostics,
                                       self.safe_platforms,self.firing_platforms,self.firing_lanes,self.combat_platforms)
                bounds=self.firing_diagnostics.pop('bounds',{})
                self.firing_bounds={q.id:(bounds[q.id][0]-q.left,bounds[q.id][1]-q.left)
                                    for q in o.platforms if q.id in bounds}
                self.firing_goal=choice[1] if choice else None
                if choice:
                    score,goal,x,path,count=choice
                    # A target already inside the firing band outranks a
                    # multi-floor reposition. In live trial
                    # run_20260927_101800, 58.5 of 323.4 s were spent starting
                    # rope/drop/jump transitions while a valid target was on
                    # screen; those frames cannot earn EXP. Travel stays
                    # available the moment the current view has no target.
                    if path and not visible:
                        self.pending_edge=path[0];self.pending_since=now
                        return self._navigate_step(o,path[0],now)
                    if not path and not visible and abs(x-p.cx)>8:
                        return Decision(frozenset({'right' if x>p.cx else 'left'}),'firing_position_align',goal)
            if quiet_wait:
                self.pending_edge=None;self.pending_since=None
                return quiet_wait
        if not floor and not self.rope_climber and not self.transition and not self.jump_combat:
            # Round 045: a drop directly over a vine left the player 18px
            # below its top. Visible enemies must not preempt rope recovery.
            recovery=self.resume_suspended_rope(o,now)
            if recovery:
                self.direct_hold_until=0;self.direct_rest_until=0;self.direct_target=None
                return recovery
        else:self.rope_recovery=None
        if (floor and abs(o.player.vy)<60 and self.rope_climber
                and self.rope_climber.phase in ('select','approach','brake')
                and (not self.sustain_farming or self.firing_goal is None
                     or any(self.melee_close(m,p) for m in o.monsters))
                and any(self.standing_target(m,p) for m in o.monsters)):
            # Preparing to catch is still grounded movement. Previously only
            # the all-jump policy could interrupt it; standing/hybrid ignored
            # visible in-range enemies for ~8 s in several short live trials.
            self.rope_climber=None;self.rope_edge=None
        # A held/slow rope pose is not standing combat. Human demonstration
        # 66–68 s keeps climbing through nearby enemies until reaching a ledge.
        # Do not abandon an active transfer at a low-velocity instant.
        if self.rope_climber or self.transition:
            self.direct_hold_until=0; self.direct_rest_until=0; self.direct_target=None
        if self.direct_attacks and not self.rope_climber and not self.transition:
            self.jump_combat=None
            if now<self.direct_hold_until:
                # A submitted hold is not evidence that its target is still
                # in front of us: live traces include knockback and crossings.
                visible=[m for m in o.monsters if self.clear_ranged_target(m,p,o.monsters)
                         and (m.box.cx>p.cx)==(self.applied_facing=='right')]
                if visible and abs(o.player.vy)<60:
                    target=min(visible,key=lambda m:abs(m.box.cx-p.cx))
                    self.direct_target=str(target.track_id)
                    self.remember_direct_target(o,target,now)
                    return Decision(frozenset({'shift'}),'direct_attack_hold',self.direct_target)
                if self.can_hold_occluded(o,now):
                    return Decision(frozenset({'shift'}),'direct_attack_occlusion_hold',self.direct_target)
                self.direct_hold_until=0;self.direct_rest_until=0;self.direct_target=None
            if now<self.direct_rest_until:
                return Decision(reason='direct_attack_complete')
            if abs(o.player.vy)<60:
                fight=self.fight(o,floor,now)
                if fight:
                    self.transition=None; self.rope_climber=None
                    return fight
        if floor and abs(o.player.vy)<60: self.last_grounded_at=now
        if self.jump_combat:
            d=self.jump_combat.decide(o,now); self.facing=self.jump_combat.facing
            if self.jump_combat.done:
                self.jump_combat=None; self.next_jump_combat=now+.25
            return d
        if (self.jump_attacks_enabled and not self.direct_attacks and floor and abs(o.player.vy)<60
                and not self.transition and not self.rope_climber
                and now>=self.next_jump_combat and (self.prefer_jump_attacks or not any(
                    self.clear_ranged_target(m,p,o.monsters) for m in o.monsters))):
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
        if abs(o.player.vy)<60 and not self.transition and not self.rope_climber:
            if self.jump_attacks_enabled and self.prefer_jump_attacks and floor and now<self.next_jump_combat:
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
                if self.rope_edge: self.fail_edge(self.rope_edge[:2],now,'rope')
                else:
                    index=self.rope_climber.rope_index
                    if index is not None and index<len(o.ropes):
                        self.rope_resume_blocked[(index,self.rope_climber.target_id)]=p.y2-o.ropes[index].top
                self.rope_climber=None; self.rope_edge=None; self.transition=None
            return d
        if self.transition:
            a,b,kind = self.transition
            if floor and floor.id not in (a,b) and abs(o.player.vy)<60:
                if o.position_quantum:
                    if self.landing_floor!=floor.id:
                        self.landing_floor=floor.id;self.landing_since=now
                    if now-self.landing_since<.18:
                        return Decision(reason='confirm_other_landing',target=floor.id)
                self.fail_edge((a,b),now,kind); self.transition=None; self.landing_since=None
                return Decision(reason='landed_elsewhere_replan')
            if floor and floor.id == b and abs(o.player.vy)<60:
                if self.landing_floor!=floor.id:self.landing_since=None;self.landing_floor=floor.id
                if self.landing_since is None: self.landing_since=now
                if now-self.landing_since<(.18 if o.position_quantum else .05): return Decision(reason='confirm_landing',target=b)
                self.verified_edges.add((a,b)); self.transition = None; self.landing_since=None
                self.facing=None
            elif now-self.transition_started > 2.5:
                self.fail_edge((a,b),now,kind); self.transition = None
                self.landing_since=None
                return Decision(reason='route_failed_replan')
            else:
                self.landing_since=None;self.landing_floor=None
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
        if not self.navigate:
            return Decision(reason='search')
        if self.sustain_farming:
            if self.safe_platforms is not None:
                from .firing_position import safe_transfer
                path=safe_transfer(o,floor,self.motion,self.safe_platforms,self.firing_platforms,
                                   self.failures,self.firing_viewport)
                if path:
                    self.firing_bounds={};self.firing_goal=path[-1][1]
                    self.pending_edge=path[0];self.pending_since=now
                    return self._navigate_step(o,path[0],now)
            # No viable firing position and no current attack: do not fall
            # through to the legacy "route to the monster's floor" policy.
            return Decision(reason='firing_position_wait')
        if (self.prefer_firing_anchor and floor.id!=self.farm_anchor
                and now>=self.anchor_cooldown_until and self.explore_until<=now
                and abs(o.player.vy)<60):
            firing_edges=graph(o.platforms,o.ropes,self.motion)
            if self.firing_anchor_available(o,firing_edges,floor):
                path=route(firing_edges,floor.id,self.farm_anchor,self.failures)
                self.pending_edge=path[0];self.pending_since=now
                return self._navigate_step(o,path[0],now)
        # Move within the observed platform towards a visible monster first.
        distant=[m for m in [*o.monsters,*o.navigation_targets] if m.confidence>=.78
                 and abs(m.box.y2-p.y2)<35 and floor.left+margin<m.box.cx<floor.right-margin]
        if distant:
            if self.navigation_stalled(p.cx,now):return Decision(reason='stuck_request_strategy')
            m=min(distant,key=lambda m:abs(m.box.cx-p.cx))
            direction='right' if m.box.cx>p.cx else 'left'
            self.facing=direction
            return Decision(frozenset({direction}), 'approach')
        edges=graph(o.platforms,o.ropes,self.motion)
        anchor_active=(self.farm_anchor is not None and now>=self.anchor_cooldown_until
                       and self.explore_until<=now and abs(o.player.vy)<60)
        if anchor_active and self.prefer_firing_anchor:
            anchor_active=self.firing_anchor_available(o,edges,floor)
        if anchor_active and floor.id==self.farm_anchor:
            # A reviewed firing ledge may be useful again after nearby monsters
            # respawn. Wait briefly; visible combat and recovery above win.
            if now-self.no_enemy_since<self.anchor_quiet_seconds:
                return Decision(reason='anchor_wait_respawn',target=self.farm_anchor)
            self.anchor_cooldown_until=now+20;anchor_active=False
        goals = [] if floor.id in self.preferred else list(self.preferred)
        if self.explore_until>now and floor.id==self.explore_origin:
            goals=list(o.minimap_goals)+[q.id for q in sorted(o.platforms,key=lambda q:(self.explored.get(q.id,0),
                    -(q.right-q.left),abs(q.y-floor.y))) if q.id!=floor.id]+goals
        elif floor.id!=self.explore_origin: self.explore_until=0
        enemy_floors=[]
        for m in o.navigation_targets:
            if m.confidence<.78:continue
            candidates=[q for q in o.platforms if q.id!=floor.id and q.left<=m.box.cx<=q.right
                        and abs(q.y-m.box.y2)<45]
            if candidates:
                q=min(candidates,key=lambda q:abs(q.y-m.box.y2))
                if q.id not in enemy_floors: enemy_floors.append(q.id)
        # The quiet search delay is for an empty view. A currently observed
        # enemy with a mapped route should not wait through that delay anew
        # after every combat interruption (13 s of round_026's 60 s trial).
        reachable_enemies=[goal for goal in enemy_floors if route(edges,floor.id,goal,self.failures)]
        if not reachable_enemies and now-self.no_enemy_since<1.5:
            return Decision(reason='search')
        if self.navigation_stalled(p.cx,now):return Decision(reason='stuck_request_strategy')
        enemy_floors=reachable_enemies
        goals=enemy_floors+goals+list(o.minimap_goals)
        # The model predicts a destination within four seconds. One-frame
        # changes must not immediately undo a short, accepted local search.
        # Combat, observed foothold/edge checks and recovery above still win.
        local_search=(self.policy_mode=='active' and self.policy_allow_stay
                      and self.policy_search_floor==floor.id and now<self.policy_search_until
                      and self.explore_until<=now)
        if self.platform_policy is not None and abs(o.player.vy)<60:
            from .demo_policy import observation_row
            predictions=self.platform_policy.predict(observation_row(o,floor))
            if predictions:
                goal,vote=predictions[0]
                stay=(self.policy_allow_stay and goal==floor.id and vote>=max(.65,self.policy_min_vote)
                      and self.motion.calibrated and floor.right-floor.left>=180 and self.explore_until<=now)
                eligible=stay or (vote>=self.policy_min_vote and goal!=floor.id and goal in edges
                          and bool(route(edges,floor.id,goal,self.failures)))
                self.policy_advice=dict(goal=goal,vote=vote,eligible=bool(eligible),mode=self.policy_mode,
                    current_floor=floor.id,top_three=predictions[:3],local_search=bool(stay))
                if eligible and self.policy_mode=='active':
                    if stay:
                        goals=[];local_search=True
                        if self.policy_search_seconds>0 and now>=self.policy_search_retry:
                            self.policy_search_floor=floor.id
                            self.policy_search_until=now+self.policy_search_seconds
                            self.policy_search_retry=now+4.
                    else:goals=[goal]+goals
        if local_search:goals=[]
        if anchor_active and floor.id!=self.farm_anchor:
            goals=[self.farm_anchor]+goals
        if not self.motion.calibrated or floor.right-floor.left<180:
            # Find room to measure motion when the spawn ledge is too small or
            # occluded; downward overlap is visible without a jump estimate.
            goals += [q.id for q in sorted(o.platforms,key=lambda p:p.right-p.left,reverse=True)
                      if q.id!=floor.id and q.right-q.left>floor.right-floor.left+50]
        for goal in goals:
            path=route(edges,floor.id,goal,self.failures)
            if path:
                self.pending_edge=path[0];self.pending_since=now
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
        return Decision(frozenset({self.facing}),'learned_platform_patrol' if local_search else 'patrol')

    def _navigate_step(self, o, edge, now, airborne=False,transit_platforms=()):
        if self.safe_platforms is not None and edge[1] not in transit_platforms:
            dest=next((q for q in o.platforms if q.id==edge[1]),None)
            occupied=dest and any(m.confidence>=.78 and dest.left-20<=m.box.cx<=dest.right+20
                and abs(m.box.y2-dest.y)<40 for m in [*o.monsters,*o.navigation_targets])
            if edge[1] not in self.safe_platforms or occupied:
                self.pending_edge=None;self.pending_since=None
                return Decision(reason='safe_platform_route_wait')
        physical_source=next((q for q in o.platforms if q.id==edge[0]),None)
        if self.sustain_farming and self.firing_goal and self.firing_bounds:
            from dataclasses import replace
            from .model import Platform
            o=replace(o,platforms=[Platform(q.id,q.left+self.firing_bounds[q.id][0],
                q.left+self.firing_bounds[q.id][1],q.y) if q.id in self.firing_bounds else q for q in o.platforms])
        a,b,kind=edge
        # Airborne traversal has its own transition deadline. It must not
        # seed an expired approach timer for the next destination.
        if self.pending_since is None and self.transition is None and not airborne:
            self.pending_since=now
        source=next((p for p in o.platforms if p.id==a),None)
        dest=next((p for p in o.platforms if p.id==b),None)
        if not source or not dest:
            self.transition=None; self.pending_edge=None; self.pending_since=None
            return Decision(reason='route_invalid')
        if (o.position_quantum and kind in ('jump','drop') and self.transition is None
                and self.standing_last_applied is not None
                and now-self.standing_last_applied<max(.35,self.attack_hold_seconds)):
            return Decision(reason='attack_finish_before_move',target=b)
        if kind=='rope':
            from .climbing import RopeClimber
            self.rope_climber=RopeClimber(target_id=dest.id,jump_height=self.motion.jump_height,speed=self.motion.speed,jump_distance=self.motion.jump_distance)
            self.rope_edge=edge
            self.pending_edge=None
            return self.rope_climber.decide(o,now)
        x=o.player.box.cx
        if kind=='drop' and self.failure_counts.get((a,b),0):
            # A monster-clear interval boundary is not a physical ledge.
            exits=[q for q,allowed in ((physical_source.left-18,abs(source.left-physical_source.left)<1),
                                      (physical_source.right+18,abs(source.right-physical_source.right)<1))
                   if allowed and dest.left+24<q<dest.right-24]
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
        if o.position_quantum and kind=='jump':
            inset=min((source.right-source.left)/3,max(32,o.position_quantum+20))
            launch=max(source.left+inset,min(source.right-inset,mid))
        overlap_jump=kind=='jump' and lo<hi
        if overlap_jump:launch=(lo+hi)/2
        if overlap_jump and o.position_quantum and hi-lo<2*o.position_quantum:
            # A one-marker-wide overlap cannot be approached precisely. Live
            # trial 11 walked off observed_16 while braking at its right edge.
            # Launch diagonally from inside the source instead; airborne
            # steering still uses the destination's inset landing interval.
            inset=min((source.right-source.left)/3,
                      max(self.motion.edge_margin,o.position_quantum+18))
            launch=max(source.left+inset,min(source.right-inset,mid))
            overlap_jump=False
        # Skip the midpoint detour only when there is room to settle. Narrow
        # overlaps retain the existing centred launch: nearest-edge braking
        # caused repeated reversals in live round_036.
        wide_drop=kind=='drop' and hi-lo>=100
        if kind=='drop' and lo<hi:
            launch=max(lo,min(hi,x)) if wide_drop else (lo+hi)/2
        drop_ropes=[r for r in o.ropes if abs(r.top-source.y)<20 and r.bottom>source.y+20]
        if kind=='drop' and self.transition is None and not airborne and drop_ropes:
            # Down at a rope mouth attaches instead of dropping through.
            # Keep the launch within the supported overlap and 32px away;
            # the approach tolerance still leaves at least 24px clearance.
            candidates=[launch,lo,hi]+[v for r in drop_ropes for v in (r.x-32,r.x+32)]
            candidates=[v for v in candidates if lo<=v<=hi and all(abs(v-r.x)>=32 for r in drop_ropes)]
            if not candidates:
                self.fail_edge((a,b),now,'drop');self.pending_edge=None;self.pending_since=None
                return Decision(reason='drop_rope_blocked',target=b)
            launch=min(candidates,key=lambda v:(abs(v-launch),v))
        if self.drop_brake_edge!=edge or self.transition is not None:
            self.drop_brake_edge=None;self.drop_brake_stable=None
        drop_settled=False
        if self.drop_brake_edge==edge and not airborne:
            relative=x-source.left
            if self.drop_brake_stable is None or abs(relative-self.drop_brake_stable[0])>5:
                self.drop_brake_stable=(relative,now)
            if now-self.drop_brake_stable[1]<.25:
                return Decision(reason='drop_brake',target=b)
            self.drop_brake_edge=None;self.drop_brake_stable=None;drop_settled=True
            # Round 067 repeatedly reversed between x=373 and x=428 even
            # though both positions were over the 347..453 landing ledge.
            # Settled support with clearance is enough; no midpoint detour.
            if (max(source.left+15,dest.left+24)<=x<=min(source.right-15,dest.right-24)
                    and all(abs(x-r.x)>=24 for r in drop_ropes)):
                launch=x
        jump_settled=False
        if self.jump_brake_edge!=edge or self.transition is not None:
            self.jump_brake_edge=None;self.jump_brake_stable=None
        if self.jump_brake_edge==edge and not airborne:
            relative=x-source.left
            # Per-frame derivatives of 1–2px pose jitter exceed 35px/s at
            # 30Hz. Observe displacement over time rather than repeatedly
            # rejecting a stationary character on one noisy velocity sample.
            if self.jump_brake_stable is None or abs(relative-self.jump_brake_stable[0])>5:
                self.jump_brake_stable=(relative,now)
            if now-self.jump_brake_stable[1]<.25:
                return Decision(reason='jump_brake',target=b)
            self.jump_brake_edge=None
            self.jump_brake_stable=None
            jump_settled=True
            # A few pixels of stopping drift inside the supported overlap
            # should not cause another walk/reverse/jump cycle.
            if max(source.left,dest.left)+8<=x<=min(source.right,dest.right)-8:launch=x
        if ((overlap_jump or (kind=='jump' and o.position_quantum)) and self.transition is None and not airborne and not jump_settled
                and abs(o.player.vx)>5 and abs(x-launch)<=max(8,abs(o.player.vx)*.20+3)):
            self.jump_brake_edge=edge;self.jump_brake_stable=None
            return Decision(reason='jump_brake',target=b)
        if (kind=='drop' and self.transition is None and not airborne and not drop_settled
                and abs(o.player.vx)>5 and abs(x-launch)<=max(8,abs(o.player.vx)*.20+3)):
            self.drop_brake_edge=edge;self.drop_brake_stable=None
            return Decision(reason='drop_brake',target=b)
        if not airborne and abs(x-launch)>max(8,o.position_quantum*.75):
            face='right' if launch>x else 'left'; self.facing=face
            return Decision(frozenset({face}), 'approach_launch',b)
        if self.transition is None:
            self.transition=edge; self.pending_edge=None; self.transition_started=now
            self.pending_since=None
            self.jump_origin_clearance=source.y-o.player.box.y2;self.jump_lift_confirmed=False
        elapsed=now-self.transition_started
        direction='right' if mid>x else 'left'
        if kind=='walk': return Decision(frozenset({direction}), 'walk_route',b)
        if kind=='drop':
            # Establish Down before the jump edge; an unordered simultaneous
            # chord can be interpreted as a normal jump by the game.
            keys={'down'} if elapsed<.40 else set()
            if .12<=elapsed<.26: keys.add('alt')
            return Decision(frozenset(keys),'drop',b)
        if o.position_quantum and not self.jump_lift_confirmed:
            rise=source.y-o.player.box.y2-self.jump_origin_clearance
            self.jump_lift_confirmed=rise>=max(8,o.position_quantum*.75)
            if not self.jump_lift_confirmed:
                if elapsed>.35:
                    self.transition=None;self.fail_edge((a,b),now,'takeoff')
                    return Decision(reason='jump_not_observed',target=b)
                # Sending Alt is not proof the game accepted a jump. Do not
                # walk off the launch platform when an attack animation blocks it.
                launch_keys={'alt'} if elapsed<.08 else set()
                if elapsed<.08 and abs(mid-x)>12:launch_keys.add(direction)
                return Decision(frozenset(launch_keys),'jump_wait_takeoff',b)
        keys={direction} if abs(mid-x)>12 else set()
        if elapsed<.080: keys.add('alt')
        return Decision(frozenset(keys), 'jump',b)


class LeasedKeys:
    """No network/model dependency; watchdog releases on timeout or focus change."""
    ALLOWED=frozenset({'left','right','up','down','shift','alt','home'})
    # Hold durations are a floor on the attack key's down time. The upper bound
    # keeps a hung control loop from holding the key indefinitely.
    HOLD_MIN=.05; HOLD_MAX=1.

    def __init__(self, adapter, hwnd, clock=time.perf_counter):
        self.adapter=adapter; self.hwnd=hwnd; self.clock=clock
        self.held=set(); self.deadline=0; self.epoch=0
        self.lock=threading.RLock(); self.stopped=threading.Event(); self.error=None
        self.thread=None; self.blocked=False
        self.pulses={}; self.release_events=[]; self.suppressed=set()
        self.holds={}; self.wanted=set()

    def allowed(self):
        return self.adapter.is_target_foreground(self.hwnd) and not self.adapter.emergency_pressed()

    def _release(self,key):
        self.adapter.send_key(key,True); self.held.discard(key)
        self.holds.pop(key,None)
        pulse=self.pulses.pop(key,None)
        if pulse:
            self.release_events.append((key,self.clock(),pulse[1])); self.suppressed.add(key)

    def _holding(self,key,now):
        """True only while a committed attack still owns the key."""
        deadline=self.holds.get(key)
        if deadline is None: return False
        if now>=deadline: self.holds.pop(key,None); return False
        return True

    def _expire_lease(self,now):
        """A late or missing frame stops new keys, not an in-flight attack hold."""
        for key in tuple(self.held):
            if not self._holding(key,now): self._release(key)

    def pop_releases(self):
        with self.lock:
            events=self.release_events; self.release_events=[]; return events

    def clear(self):
        with self.lock:
            self.holds.clear(); self.wanted.clear()
            errors=[]
            for key in tuple(self.held):
                try: self._release(key)
                except Exception as e: errors.append(e)
            if errors: raise RuntimeError('Failed to release one or more keys') from errors[0]

    def check(self):
        with self.lock:
            now=self.clock()
            allowed=self.allowed()
            if not allowed and not self.blocked: self.epoch+=1
            self.blocked=not allowed
            if self.adapter.emergency_pressed(): self.stopped.set()
            if not allowed or self.stopped.is_set(): self.clear()
            elif now>=self.deadline: self._expire_lease(now)
            for key,(at,token) in list(self.pulses.items()):
                if now>=at and key in self.held: self._release(key)
            for key in tuple(self.held):
                # A held key whose owner stopped asking for it is released as
                # soon as its own hold elapses, even with a live lease.
                if key in self.wanted or key in self.pulses: continue
                if not self._holding(key,now): self._release(key)

    def apply(self, keys, captured_at, epoch, pulses=None, holds=None):
        with self.lock:
            now=self.clock(); keys=set(keys)
            if not keys<=self.ALLOWED or {'left','right'}<=keys or {'up','down'}<=keys:
                self.clear(); raise ValueError('Invalid key state')
            if holds:
                for key,duration in holds.items():
                    if key not in keys or not self.HOLD_MIN<=duration<=self.HOLD_MAX:
                        raise ValueError('Invalid hold')
            if self.stopped.is_set() or epoch!=self.epoch or not self.allowed():
                self.clear(); return False
            if not 0<=now-captured_at<.085:
                # Old and future frames may not add or re-press any key, but a
                # committed attack keeps its hold instead of being cut short.
                self._expire_lease(now); return False
            self.suppressed.intersection_update(keys)
            if pulses:
                for key,(duration,token) in pulses.items():
                    if key not in keys or not 0<duration<=.08: raise ValueError("Invalid pulse")
                    self.suppressed.discard(key)
            keys-=self.suppressed
            # Frame age plus lease never exceeds 100ms from capture start.
            self.deadline=min(captured_at+.095,now+.060)
            keep={key for key in self.held-keys if self._holding(key,now)}
            for key in tuple(self.held-keys-keep): self._release(key)
            self.held.intersection_update(keys|keep)
            for key in sorted(keys-self.held,key=lambda k:(k in ('alt','shift','home'),k)):
                self.held.add(key)
                try: self.adapter.send_key(key,False)
                except Exception:
                    self.stopped.set(); self.clear(); raise
            for key,(duration,token) in (pulses or {}).items():
                self.pulses[key]=(self.clock()+duration,token)
            for key,duration in (holds or {}).items():
                self.holds[key]=self.clock()+duration
            self.wanted=set(keys)
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
