"""Closed-loop rope acquisition, ascent, and landing. No timed blind macro."""
from .control import standing_platform
from .model import Decision


class RopeClimber:
    def __init__(self,target_id=None,jump_height=90,speed=180,jump_distance=None):
        self.requested_target=target_id; self.jump_height=jump_height
        self.speed=speed; self.jump_distance=jump_distance if jump_distance is not None else speed*.5
        self.catch_direction=None
        self.start_relative=None; self.latch_since=None; self.grab_confirmed=False
        self.phase='select'; self.target_id=None; self.rope_index=None
        self.phase_at=0; self.started=0; self.attempts=0; self.done=False
        self.last_epoch=None; self.start_y=None; self.preferred=[]
        self.verified_edges=set(); self.failures=set(); self.facing=None
        self.landed_since=None

    def reset(self):
        self.phase='select'; self.phase_at=0; self.started=0; self.attempts=0
        self.target_id=None; self.rope_index=None; self.start_y=None
        self.landed_since=None
        self.done=False
        self.catch_direction=None
        self.start_relative=None; self.latch_since=None; self.grab_confirmed=False

    def decide(self,o,now):
        if self.last_epoch is None: self.last_epoch=o.map_epoch
        if o.map_epoch!=self.last_epoch:
            self.reset(); self.last_epoch=o.map_epoch
        if self.done: return Decision(reason='climb_complete')
        if now-o.captured_at>=.085: return Decision(reason='stale_frame')
        if o.reason or not o.player or not o.motion_valid:
            return Decision(reason=o.reason or 'climb_wait_for_player')
        player=o.player.box; floor=standing_platform(o)
        # Contact with the destination takes precedence over horizontal rope
        # alignment: monsters may knock the character sideways on arrival.
        if (self.attempts and floor and floor.id==self.target_id and abs(o.player.vy)<60
                and abs(player.y2-floor.y)<9):
            if self.landed_since is None: self.landed_since=now
            if now-self.landed_since>=.05:
                self.done=True; return Decision(reason='climb_complete',target=self.target_id)
            return Decision(reason='rope_confirm_landing',target=self.target_id)
        self.landed_since=None
        if self.phase=='failed': return Decision(reason='climb_failed_request_strategy')
        if self.started and now-self.started>8:
            self.phase='failed'; return Decision(reason='climb_timeout')
        if self.phase=='select':
            if floor is None: return Decision(reason='climb_wait_for_floor')
            choices=[]
            for i,r in enumerate(o.ropes):
                # A rope can end above the floor; a short jump may be needed.
                if not floor.left+12<r.x<floor.right-12 or not -15<=floor.y-r.bottom<=max(0,self.jump_height-12): continue
                for p in o.platforms:
                    if p.id==floor.id or p.y>=floor.y-30: continue
                    if self.requested_target and p.id!=self.requested_target: continue
                    if p.left+20<r.x<p.right-20 and abs(p.y-r.top)<15:
                        choices.append((0 if p.id in self.preferred else 1,abs(r.x-player.cx),i,p.id))
            if not choices:
                self.phase='failed'; return Decision(reason='no_accessible_rope')
            _,_,self.rope_index,self.target_id=min(choices)
            self.started=now; self.phase_at=now; self.phase='approach'
        if self.rope_index>=len(o.ropes): self.reset(); return Decision(reason='rope_map_invalid')
        rope=o.ropes[self.rope_index]
        target=next((p for p in o.platforms if p.id==self.target_id),None)
        if target is None: self.reset(); return Decision(reason='rope_target_missing')
        dx=rope.x-player.cx
        if self.phase=='approach':
            if floor is None: return Decision(reason='climb_wait_for_floor')
            # Intercept the rope in the air: maintain Up throughout the
            # diagonal jump, and stop lateral input when reaching its x.
            vertical_gap=max(0,floor.y-rope.bottom)
            if vertical_gap>max(0,self.jump_height-12):
                self.phase="failed"; return Decision(reason="rope_out_of_reach")
            # Reach shrinks as the rope end approaches the jump apex.
            height_ratio=min(1,vertical_gap/max(1,self.jump_height))
            reach=self.jump_distance*.45*(1-height_ratio)**.5
            launch_distance=max(7,min(45,self.speed*.20,reach))
            moving_toward=o.player.vx*dx>=0 or abs(o.player.vx)<35
            if 18<abs(dx)<=launch_distance and moving_toward:
                self.phase='catch'; self.phase_at=now; self.start_y=player.y2
                self.attempts+=1; self.catch_direction='right' if dx>0 else 'left'
            # Jumping directly from a run preserves horizontal momentum. Brake
            # near the rope and measure that motion has settled before jumping.
            elif abs(dx)<=max(9,abs(o.player.vx)*.16) and abs(o.player.vx)>35:
                self.phase='brake'; self.phase_at=now
                return Decision(reason='rope_brake',target=self.target_id)
            elif abs(dx)>7:
                return Decision(frozenset({'right' if dx>0 else 'left'}),'rope_approach',self.target_id)
            else:
                self.phase='catch'; self.phase_at=now; self.start_y=player.y2; self.attempts+=1
                self.catch_direction=None
        if self.phase=='brake':
            if not floor or now-self.phase_at<.15 or abs(o.player.vx)>35:
                return Decision(reason='rope_brake',target=self.target_id)
            if abs(dx)>7:
                self.phase='approach'; return Decision(reason='rope_align',target=self.target_id)
            self.phase='catch'; self.phase_at=now; self.start_y=player.y2; self.attempts+=1
            self.catch_direction=None
        if self.phase=='catch':
            elapsed=now-self.phase_at
            if self.start_relative is None: self.start_relative=self.start_y-rope.top
            rise=self.start_relative-(player.y2-rope.top)
            if elapsed>.3 and abs(dx)>75:
                self.phase='approach'; self.start_relative=None; self.latch_since=None
                return Decision(reason='rope_missed_reposition')
            # A normal diagonal jump must not be reported as grabbing a rope.
            # Use camera-independent height and sustained rope alignment.
            if rise>self.jump_height+12 and abs(dx)<12:
                if self.latch_since is None: self.latch_since=now
            else: self.latch_since=None
            if self.latch_since is not None and now-self.latch_since>=.12:
                self.grab_confirmed=True; self.phase='ascend'; self.phase_at=now
            elif elapsed>1.25:
                if self.attempts>=3: self.phase='failed'; return Decision(reason='rope_catch_failed')
                self.phase='approach'; self.start_relative=None; self.latch_since=None
                return Decision(reason='rope_retry')
            else:
                keys={'up'}
                if elapsed<.08: keys.add('alt')
                if self.catch_direction:
                    approaching=(dx>0)==(self.catch_direction=='right')
                    if approaching and abs(dx)>max(7,abs(o.player.vx)*.04):
                        keys.add(self.catch_direction)
                    else: self.catch_direction=None
                return Decision(frozenset(keys),'rope_catch_diagonal' if self.catch_direction else 'rope_catch',self.target_id)
        if self.phase=='ascend':
            if abs(dx)>25:
                self.phase='approach'; return Decision(reason='rope_lost')
            if player.y2<=target.y+3:
                self.phase='confirm'; self.phase_at=now
            return Decision(frozenset({'up'}),'rope_ascend',self.target_id)
        if self.phase=='confirm':
            if now-self.phase_at<.12: return Decision(frozenset({'up'}),'rope_step_off',self.target_id)
            if floor and floor.id==target.id and abs(player.y2-target.y)<14:
                self.done=True; return Decision(reason='climb_complete',target=self.target_id)
            if now-self.phase_at>.7:
                self.phase='ascend'; self.phase_at=now
            return Decision(reason='rope_confirm_landing',target=self.target_id)
        return Decision(reason='climb_wait')
