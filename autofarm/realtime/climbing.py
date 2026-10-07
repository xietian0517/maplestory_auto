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
        self.source_id=None; self.wrong_floor_since=None
        self.braked_since=None
        self.probe_relative=None; self.progress_relative=None; self.progress_at=None

    def reset(self):
        self.phase='select'; self.phase_at=0; self.started=0; self.attempts=0
        self.target_id=None; self.rope_index=None; self.start_y=None
        self.landed_since=None
        self.source_id=None; self.wrong_floor_since=None
        self.braked_since=None
        self.probe_relative=None; self.progress_relative=None; self.progress_at=None
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
                and abs(player.y2-floor.y)<max(9,o.position_quantum)):
            if self.landed_since is None: self.landed_since=now
            if now-self.landed_since>=.05:
                self.done=True; return Decision(reason='climb_complete',target=self.target_id)
            return Decision(reason='rope_confirm_landing',target=self.target_id)
        self.landed_since=None
        # Falling onto another ledge ends this transfer. Never reuse its
        # alignment and jump sequence from a different launch platform.
        if (self.source_id and floor and floor.id not in (self.source_id,self.target_id)
                and abs(o.player.vy)<60):
            if self.wrong_floor_since is None: self.wrong_floor_since=now
            if now-self.wrong_floor_since>=.12:
                self.phase='failed'
                return Decision(reason='rope_landed_elsewhere',target=self.target_id)
            return Decision(reason='rope_confirm_wrong_floor',target=self.target_id)
        self.wrong_floor_since=None
        if self.phase=='failed': return Decision(reason='climb_failed_request_strategy')
        if self.started and now-self.started>(40 if o.position_quantum else 8):
            self.phase='failed'; return Decision(reason='climb_timeout')
        if self.phase=='select':
            if floor is None: return Decision(reason='climb_wait_for_floor')
            choices=[]
            for i,r in enumerate(o.ropes):
                # A rope can end above the floor; a short jump may be needed.
                if not floor.left+12<r.x<floor.right-12 or floor.y-r.bottom>max(0,self.jump_height-12): continue
                for p in o.platforms:
                    if p.id==floor.id or p.y>=floor.y-30: continue
                    if self.requested_target and p.id!=self.requested_target: continue
                    if p.left+20<r.x<p.right-20 and abs(p.y-r.top)<15:
                        choices.append((0 if p.id in self.preferred else 1,abs(r.x-player.cx),i,p.id))
            if not choices:
                self.phase='failed'; return Decision(reason='no_accessible_rope')
            _,_,self.rope_index,self.target_id=min(choices)
            self.source_id=floor.id
            self.started=now; self.phase_at=now; self.phase='approach'
        if self.rope_index>=len(o.ropes): self.reset(); return Decision(reason='rope_map_invalid')
        rope=o.ropes[self.rope_index]
        target=next((p for p in o.platforms if p.id==self.target_id),None)
        if target is None: self.reset(); return Decision(reason='rope_target_missing')
        dx=rope.x-player.cx
        if self.phase in ('approach','brake','probe') and rope.top>=player.y2-12 and target.y>=player.y2-12:
            self.phase='failed'; return Decision(reason='rope_not_above_player',target=self.target_id)
        if self.phase=='probe':
            relative=player.y2-rope.top
            if self.probe_relative is None: self.probe_relative=relative
            if abs(dx)>12:
                self.phase='failed'; return Decision(reason='rope_resume_unconfirmed',target=self.target_id)
            if self.probe_relative-relative>=8 and now-self.phase_at>=.12:
                self.grab_confirmed=True; self.phase='ascend'; self.progress_relative=relative; self.progress_at=now
            elif now-self.phase_at>.65:
                self.phase='failed'; return Decision(reason='rope_resume_unconfirmed',target=self.target_id)
            else: return Decision(frozenset({'up'}),'rope_probe_attachment',self.target_id)
        alignment_dx=dx+(0,4,-4)[min(self.attempts,2)]
        alignment_tolerance=max(3 if self.attempts==0 else 2,o.position_quantum*.6)
        if self.phase=='approach':
            if floor is None: return Decision(reason='climb_wait_for_floor')
            # First stop under the rope. Releasing a direction in mid-air does
            # not remove horizontal momentum (confirmed by captured failures).
            vertical_gap=max(0,floor.y-rope.bottom)
            if vertical_gap>max(0,self.jump_height-12):
                self.phase="failed"; return Decision(reason="rope_out_of_reach")
            if abs(alignment_dx)<=max(9,abs(o.player.vx)*.20+3) and abs(o.player.vx)>20:
                self.phase='brake'; self.phase_at=now; self.braked_since=None
                return Decision(reason='rope_brake',target=self.target_id)
            elif abs(alignment_dx)>alignment_tolerance:
                if abs(alignment_dx)<=12 and abs(o.player.vx)<=20:
                    # Live traces show a delayed 20–35px overshoot when a
                    # final correction waits for measured velocity before
                    # releasing. Restrict this to the last 12px: round_028 took
                    # 15 tiny nudges to cover 38px with the former 45px band.
                    # Use one observed control tick, then settle.
                    self.phase='brake';self.phase_at=now;self.braked_since=None
                    return Decision(frozenset({'right' if alignment_dx>0 else 'left'}),'rope_fine_nudge',self.target_id)
                return Decision(frozenset({'right' if alignment_dx>0 else 'left'}),'rope_approach',self.target_id)
            else:
                self.phase='catch'; self.phase_at=now; self.start_y=player.y2; self.attempts+=1
                self.catch_direction=None
        if self.phase=='brake':
            if not floor or abs(o.player.vx)>20 or abs(o.player.vy)>60:
                self.braked_since=None
                return Decision(reason='rope_brake',target=self.target_id)
            if self.braked_since is None: self.braked_since=now
            if now-self.braked_since<.12-1e-6:
                return Decision(reason='rope_brake',target=self.target_id)
            if abs(alignment_dx)>alignment_tolerance:
                self.phase='approach'; return Decision(reason='rope_align',target=self.target_id)
            self.phase='catch'; self.phase_at=now; self.start_y=player.y2; self.attempts+=1
            self.catch_direction=None
        if self.phase=='catch':
            elapsed=now-self.phase_at
            if self.start_relative is None: self.start_relative=self.start_y-rope.top
            rise=self.start_relative-(player.y2-rope.top)
            if elapsed>.3 and abs(dx)>35:
                return self.retry(now,'rope_missed_reposition')
            # A normal diagonal jump must not be reported as grabbing a rope.
            # Use camera-independent height and sustained rope alignment.
            if rise>self.jump_height+12 and abs(dx)<12:
                if self.latch_since is None: self.latch_since=now
            else: self.latch_since=None
            if self.latch_since is not None and now-self.latch_since>=.12:
                self.grab_confirmed=True; self.phase='ascend'; self.phase_at=now
            elif elapsed>1.25:
                # Recorded 26.7 s: late attachment was already rising, but
                # had not exceeded a full jump's height. Releasing Up here
                # stranded the character on the rope. Confirm progress with
                # the same bounded probe used for suspended-rope recovery.
                if floor is None and abs(dx)<12 and rope.top<player.y2<=rope.bottom+12:
                    self.phase='probe';self.phase_at=now
                    self.probe_relative=player.y2-rope.top
                    return Decision(frozenset({'up'}),'rope_probe_attachment',self.target_id)
                return self.retry(now,'rope_retry')
            else:
                keys={'up'}
                if elapsed<.08: keys.add('alt')
                if self.catch_direction:
                    approaching=(dx>0)==(self.catch_direction=='right')
                    if approaching and abs(dx)>max(7,abs(o.player.vx)*.04):
                        keys.add(self.catch_direction)
                    else: self.catch_direction=None
                return Decision(frozenset(keys),'rope_catch_diagonal' if self.catch_direction else 'rope_catch',self.target_id)
        if self.phase=='recover':
            # Wait for an observed landing before another alignment/jump. Do not
            # steer left/right while falling past the rope after a missed catch.
            if floor and abs(o.player.vy)<60:
                self.phase='approach'; self.braked_since=None
            return Decision(reason='rope_wait_for_landing',target=self.target_id)
        if self.phase=='ascend':
            if abs(dx)>25:
                self.phase='approach'; return Decision(reason='rope_lost')
            if player.y2<=target.y+3:
                self.phase='confirm'; self.phase_at=now
            relative=player.y2-rope.top
            if self.progress_relative is None or relative<self.progress_relative-4:
                self.progress_relative=relative; self.progress_at=now
            elif self.phase=='ascend' and now-self.progress_at>.8:
                self.phase='failed'; return Decision(reason='rope_no_vertical_progress',target=self.target_id)
            return Decision(frozenset({'up'}),'rope_ascend',self.target_id)
        if self.phase=='confirm':
            if now-self.phase_at<.12: return Decision(frozenset({'up'}),'rope_step_off',self.target_id)
            # Only the sustained, low-velocity landing check above may complete.
            if now-self.phase_at>.7:
                self.phase='ascend'; self.phase_at=now
            return Decision(reason='rope_confirm_landing',target=self.target_id)
        return Decision(reason='climb_wait')

    def retry(self,now,reason):
        self.start_relative=None; self.latch_since=None; self.catch_direction=None
        self.grab_confirmed=False; self.phase_at=now
        if self.attempts>=3:
            self.phase='failed'; return Decision(reason='rope_catch_failed',target=self.target_id)
        self.phase='recover'
        return Decision(reason=reason,target=self.target_id)
