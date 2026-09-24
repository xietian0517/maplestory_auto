"""Short observed recovery actions independent of route-planner waiting states."""
from .model import Decision


class ActiveRecovery:
    waits={'search','airborne_or_floor_unknown','climb_wait_for_floor','approach_wait_for_floor',
           'player_not_found','identity_confirming','route_retry_pending','stuck_request_strategy',
           'route_failed_replan','launch_approach_timeout','climb_failed_request_strategy'}

    def __init__(self): self.reset()

    def reset(self):
        self.idle_since=None; self.last_x=None; self.position=None; self.progress_at=None
        self.until=0; self.action=None; self.attempt=0; self.epoch=None
        self.last_engagement=None; self.last_replan=None

    def apply(self,o,d,now,width,offset=(0,0),controller=None):
        if o is None or not o.motion_valid or now-o.captured_at>=.085:
            return d
        if self.epoch!=o.map_epoch: self.reset(); self.epoch=o.map_epoch
        if self.last_engagement is None: self.last_engagement=now; self.last_replan=now
        if o.player and not o.reason:
            self.last_x=o.player.box.cx
            pos=(o.player.box.cx-offset[0],o.player.box.y2-offset[1])
            if self.position is None or abs(pos[0]-self.position[0])+abs(pos[1]-self.position[1])>12:
                self.position=pos; self.progress_at=now
        # Combat always interrupts recovery, and receives its normal priority.
        if 'shift' in d.keys:
            self.last_engagement=now
            self.idle_since=None; self.until=0; self.progress_at=now
            return d
        if controller is not None and now-max(self.last_engagement,self.last_replan)>6:
            base=getattr(controller,'base',controller)
            if base.request_exploration(o,now):
                self.last_replan=now; self.until=0
                return Decision(reason='engagement_timeout_replan')
        if now<self.until: return self.action
        settling=d.reason.startswith(('calibrat','jump_attack_')) or d.reason in ('rope_brake','rope_confirm_landing','confirm_landing')
        airborne=o.player is not None and abs(o.player.vy)>80
        waiting=not d.keys and not settling and not airborne
        stuck=bool(d.keys) and self.progress_at is not None and now-self.progress_at>3
        if not waiting and not stuck:
            self.idle_since=None
            return d
        if self.idle_since is None: self.idle_since=now
        if not stuck and now-self.idle_since<.8: return d
        if self.last_x is None: return d
        direction='right' if self.last_x<width/2 else 'left'
        if self.attempt%3==1: direction='left' if direction=='right' else 'right'
        keys={direction}
        if self.attempt%3==2: keys.add('alt')
        self.attempt+=1; self.until=now+.18; self.idle_since=None; self.progress_at=now
        self.action=Decision(frozenset(keys),'active_recovery_step')
        return self.action
