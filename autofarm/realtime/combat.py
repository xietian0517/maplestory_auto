"""Observed jump attacks with two acknowledged, separated attack presses."""
from .model import Decision


class JumpAttack:
    EARLY_JUMP_HOLD=.040
    EARLY_RELEASE_GAP=.030
    EARLY_FIRE_WINDOW=.180

    def __init__(self,floor,target,now,facing,attack_max):
        self.floor_id=floor.id; self.target_id=target.track_id; self.shot_target_id=None
        self.started=now; self.jumped_at=None; self.jump_released_at=None; self.facing=facing
        self.attack_max=attack_max; self.done=False; self.turn_until=now
        self.shots_issued=0; self.press_until=0; self.next_shot_at=0
        self.jump_count=0; self.shot_this_jump=False; self.next_jump_at=0; self.target_missing_at=None

    def on_input_applied(self,d,now):
        if d.reason=='jump_attack_takeoff' and self.jumped_at is None:
            self.jumped_at=now; self.jump_released_at=None; self.jump_count+=1; self.shot_this_jump=False
        if self.jumped_at is not None and 'alt' not in d.keys and self.jump_released_at is None:
            self.jump_released_at=now
        if d.reason in ('jump_attack_fire_first','jump_attack_fire_second') and not self.shot_this_jump:
            self.shots_issued+=1; self.shot_this_jump=True; self.shot_target_id=d.target
            self.press_until=now+.24; self.next_shot_at=now+.55

    def decide(self,o,now):
        if o.reason or not o.player or not o.motion_valid or now-o.captured_at>=.085:
            return Decision(reason=o.reason or 'jump_attack_wait_frame')
        p=o.player.box
        source=next((q for q in o.platforms if q.id==self.floor_id),None)
        if now-self.started>3.5 or source is None:
            self.done=True; return Decision(reason='jump_attack_finished')
        candidates=[m for m in o.monsters if m.confidence>=.78 and abs(m.box.cx-p.cx)<self.attack_max
                    and abs(m.box.y2-p.y2)<120]
        nearest=min(candidates,key=lambda m:abs(m.box.cx-p.cx)) if candidates else None
        tracked=next((m for m in candidates if m.track_id==self.target_id),None)
        target=tracked if tracked and abs(tracked.box.cx-p.cx)<=abs(nearest.box.cx-p.cx)+60 else nearest
        if target: self.target_id=target.track_id
        if target is None:
            if self.target_missing_at is None: self.target_missing_at=now
            if now-self.target_missing_at>.30: self.done=True
            return Decision(reason='jump_attack_target_lost' if self.done else 'jump_attack_target_occluded')
        self.target_missing_at=None
        if self.jumped_at is None:
            if now<max(self.next_jump_at,self.next_shot_at): return Decision(reason='jump_attack_cooldown')
            if abs(o.player.vx)>60: return Decision(reason='jump_attack_brake')
            face='right' if target.box.cx>=p.cx else 'left'
            if face!=self.facing: self.facing=face; self.turn_until=now+.04
            if now<self.turn_until: return Decision(frozenset({face}),'jump_attack_turn')
            self.early_jump=abs(source.y-target.box.y2)<35
            return Decision(frozenset({'alt'}),'jump_attack_takeoff')
        elapsed=now-self.jumped_at; rise=source.y-p.y2
        landed=next((q for q in o.platforms if q.left<=p.cx<=q.right and abs(q.y-p.y2)<12),None)
        if elapsed>.28 and abs(o.player.vy)<60 and landed:
            if self.shots_issued>=2 or self.jump_count>=3:
                self.done=True; return Decision(reason='jump_attack_landed')
            self.floor_id=landed.id; self.jumped_at=None; self.next_jump_at=now+.08
            return Decision(reason='jump_attack_followup_jump')
        height=source.y-target.box.y2
        early=abs(height)<35
        hold=self.EARLY_JUMP_HOLD if early else .08
        keys={'alt'} if elapsed<hold else set()
        # Same-level targets need a short jump-to-fire gap. Higher targets wait
        # for measured ascent into their firing band, not a fixed long sleep.
        close=abs(target.box.cx-p.cx)<65
        minimum_rise=max(15 if close else 6,height-30)
        airborne=rise>=minimum_rise and (o.player.vy<-30 or rise>=20)
        aligned=abs(target.box.y2-p.y2)<35
        face='right' if target.box.cx>=p.cx else 'left'
        if face!=self.facing: self.facing=face; self.turn_until=now+.04
        if now<self.turn_until:
            keys.add(face); return Decision(frozenset(keys),'jump_attack_turn')
        if self.shot_this_jump and now<self.press_until and str(target.track_id)==self.shot_target_id:
            keys.add('shift'); return Decision(frozenset(keys),'jump_attack_hold',str(target.track_id))
        # random_jump: release Alt, then wait 30ms before an early attack.
        # This path does not wait for a delayed screenshot to prove takeoff.
        early_ready=(early and elapsed>=self.EARLY_JUMP_HOLD+self.EARLY_RELEASE_GAP
                     and self.jump_released_at is not None
                     and now-self.jump_released_at>=self.EARLY_RELEASE_GAP
                     and elapsed<=self.EARLY_FIRE_WINDOW and -12<=rise<=60)
        # A delayed but still aligned target may receive a recovery shot. Do
        # not wait an entire jump after narrowly missing the early deadline.
        recovery_ready=(early and self.EARLY_FIRE_WINDOW<elapsed<=.32 and aligned
                        and -12<=rise<=45 and self.jump_released_at is not None
                        and now-self.jump_released_at>=self.EARLY_RELEASE_GAP)
        ready=(early_ready or recovery_ready) if early else airborne and aligned
        if ready and not self.shot_this_jump and now>=self.next_shot_at:
            self.fire_mode=('early_recovery' if recovery_ready else 'early') if early else 'upper'
            keys.add('shift')
            return Decision(frozenset(keys),'jump_attack_fire_first' if self.shots_issued==0 else 'jump_attack_fire_second',str(target.track_id))
        return Decision(frozenset(keys),'jump_attack_rising' if keys else 'jump_attack_align_height')


class HasteRefresh:
    """Refresh on stable ground; only successful input advances the timer."""
    def __init__(self): self.next_at=0; self.hold_until=0

    def apply(self,o,d,now,controller):
        from .control import standing_platform
        base=getattr(controller,'base',controller)
        if not o or o.reason or not o.player or not o.motion_valid or now-o.captured_at>=.085: return d
        if now<self.hold_until: return Decision(frozenset({'home'}),'haste_hold')
        jump=getattr(base,'jump_combat',None); rope=getattr(base,'rope_climber',None)
        if (now>=self.next_at and standing_platform(o) and abs(o.player.vy)<60
                and (jump is None or jump.jumped_at is None)
                and (rope is None or rope.phase in ('select','approach','brake'))
                and not getattr(base,'transition',None)):
            return Decision(frozenset({'home'}),'haste_refresh')
        return d

    def on_input_applied(self,d,now):
        if d.reason=='haste_refresh': self.next_at=now+120; self.hold_until=now+.10
