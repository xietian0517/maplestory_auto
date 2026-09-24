"""Measure movement and a vertical jump before proposing cross-platform jumps."""
from .control import Controller, standing_platform
from .model import Decision, MotionProfile
from statistics import median


class Calibrator:
    def __init__(self):
        self.phase='start'; self.at=0; self.origin=None; self.floor_id=None
        self.direction='right'; self.duration=.06; self.speed=0
        self.peak=0; self.peak_at=0; self.jump_at=0; self.result=None
        self.failed=False; self.last_epoch=None; self.landing_at=None

    def decide(self,o,now):
        if self.failed: return Decision(reason='calibration_failed')
        if o.reason or not o.player or not o.motion_valid: return Decision(reason=o.reason or 'calibration_wait')
        if now-o.captured_at>=.085: return Decision(reason='stale_frame')
        if self.last_epoch is None: self.last_epoch=o.map_epoch
        if self.last_epoch!=o.map_epoch: self.failed=True; return Decision(reason='calibration_map_changed')
        floor=standing_platform(o); player=o.player.box
        if self.phase=='start':
            if not floor or floor.right-floor.left<110: return Decision(reason='calibration_wait_for_room')
            left,right=player.cx-floor.left,floor.right-player.cx
            room=max(left,right)
            if room<65: return Decision(reason='calibration_wait_for_room')
            self.direction='right' if right>=left else 'left'
            self.duration=min(.25,(room-30)/600)
            self.floor_id=floor.id; self.origin=player.cx-floor.left
            self.at=now; self.phase='walk'
        base=next((p for p in o.platforms if p.id==self.floor_id),None)
        if base is None: self.failed=True; return Decision(reason='calibration_floor_lost')
        if self.phase=='walk':
            if now-self.at<self.duration: return Decision(frozenset({self.direction}),'calibrate_walk')
            self.walk_end=now; self.phase='settle'
            return Decision(reason='calibrate_settle')
        if self.phase=='settle':
            if now-self.walk_end<.12: return Decision(reason='calibrate_settle')
            distance=abs((player.cx-base.left)-self.origin)
            self.speed=distance/max(.03,self.walk_end-self.at)
            if not 60<=self.speed<=650:
                self.failed=True; return Decision(reason='calibration_walk_unreliable')
            self.phase='return'; self.at=now
        if self.phase=='return':
            dx=self.origin-(player.cx-base.left)
            if abs(dx)>7 and now-self.at<.7:
                return Decision(frozenset({'right' if dx>0 else 'left'}),'calibrate_return')
            self.phase='prepare_jump'; self.at=now
        if self.phase=='prepare_jump':
            if now-self.at<.15: return Decision(reason='calibrate_settle')
            self.phase='jump'; self.jump_at=now; self.peak_at=now
        if self.phase=='jump':
            elapsed=now-self.jump_at; height=base.y-player.y2
            if height>self.peak: self.peak=height; self.peak_at=now
            landed=elapsed>.35 and floor and abs(o.player.vy)<60 and self.peak>25
            if landed:
                if self.landing_at is None: self.landing_at=now
            else: self.landing_at=None
            if landed and (floor.id==self.floor_id or now-self.landing_at>=.08):
                # Horizontal reach is a conservative kinematic estimate; actual
                # graph edges remain unverified until successfully traversed.
                flight=max(.25,2*(self.peak_at-self.jump_at))
                self.result=MotionProfile(min(600,max(60,self.speed)),self.peak,
                                          min(500,self.speed*flight*.80),True)
                self.phase='done'; return Decision(reason='calibration_complete')
            if elapsed>2.5:
                self.failed=True; return Decision(reason='calibration_jump_unreliable')
            return Decision(frozenset({'alt'}) if elapsed<.08 else frozenset(),'calibrate_jump')
        return Decision(reason='calibration_complete')


class NavigationController:
    def __init__(self,motion=None):
        self.base=Controller(motion,True)
        self.calibrator=None
        self.preferred=[]; self.result=motion; self.started=False
        self.failed_floors=set()
        self.last_epoch=None
        self.travel_last=None; self.travel_samples=[]; self.travel_started=None; self.last_action=None

    @property
    def verified_edges(self): return self.base.verified_edges
    @property
    def failures(self): return self.base.failures

    def reset(self):
        self.base.reset()
        self.travel_last=None; self.travel_samples=[]; self.travel_started=None; self.last_action=None
        if self.calibrator: self.calibrator=Calibrator()

    def decide(self,o,now):
        self.base.preferred=self.preferred
        if self.last_epoch!=o.map_epoch:
            self.reset(); self.failed_floors.clear(); self.last_epoch=o.map_epoch
        floor=standing_platform(o)
        if self.base.jump_combat:
            self.last_action=self.base.decide(o,now)
            return self.last_action
        if (o.player and abs(o.player.vy)<60 and any(m.confidence>=.78
                and abs(m.box.y2-o.player.box.y2)<35
                and abs(m.box.cx-o.player.box.cx)<self.base.motion.attack_max for m in o.monsters)):
            self.calibrator=None
            self.travel_last=None; self.travel_samples=[]; self.travel_started=None
            self.last_action=self.base.decide(o,now)
            return self.last_action
        if (not self.calibrator and not self.base.motion.calibrated and floor
                and floor.id not in self.failed_floors and floor.right-floor.left>=110
                and not o.monsters and not self.base.transition and not self.base.rope_climber):
            self.calibrator=Calibrator()
        if self.calibrator:
            d=self.calibrator.decide(o,now)
            if self.calibrator.result:
                self.result=self.calibrator.result; self.base.motion=self.result; self.calibrator=None
                self.base.reset()
            elif self.calibrator.failed:
                # Same-platform combat remains available; no invented physics.
                self.failed_floors.add(self.calibrator.floor_id)
                self.calibrator=None
            return d
        # A short calibration walk can measure acceleration instead of cruising
        # speed. Refine reach from observed same-floor travel, never from a key
        # duration alone. Keep displacement relative to the platform for scroll.
        direction=next(iter(self.last_action.keys),None) if self.last_action and len(self.last_action.keys)==1 else None
        if (floor and o.player and not o.reason and o.motion_valid and now-o.captured_at<.085
                and abs(o.player.vy)<60 and self.base.motion.calibrated and direction in ('left','right')):
            current=(floor.id,o.player.box.cx-floor.left,now,direction)
            if self.travel_last:
                ident,x,t,key=self.travel_last;dt=now-t
                displacement=(current[1]-x)*(1 if direction=='right' else -1)
                if ident==floor.id and key==direction and .015<=dt<=.15 and 60<=displacement/dt<=500:
                    if self.travel_started is None: self.travel_started=t
                    self.travel_samples.append(displacement/dt); self.travel_samples=self.travel_samples[-10:]
                    if len(self.travel_samples)>=6:
                        speed=median(self.travel_samples)
                        deviation=median(abs(x-speed) for x in self.travel_samples)
                        # Short starts and turnarounds do not measure cruising.
                        duration=now-self.travel_started
                        minimum=.35 if speed>self.base.motion.speed else .75
                        if duration>=minimum and deviation<speed*.25 and abs(speed-self.base.motion.speed)>self.base.motion.speed*.2:
                            scale=speed/self.base.motion.speed
                            self.base.motion.jump_distance=min(500,self.base.motion.jump_distance*scale)
                            self.base.motion.speed=speed; self.result=self.base.motion
                else: self.travel_samples=[]; self.travel_started=None
            self.travel_last=current
        else: self.travel_last=None; self.travel_samples=[]; self.travel_started=None
        self.last_action=self.base.decide(o,now)
        return self.last_action
