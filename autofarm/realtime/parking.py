"""Return to an explicitly reviewed refuge, then verify a quiet stationary hold."""
from dataclasses import replace

import numpy as np

from .control import Controller, graph, route, standing_platform
from .model import Decision


def validate_parking(data,scene):
    if data.get('request_id')!=scene.request_id:
        raise ValueError('Parking plan belongs to another scene request')
    ids=data.get('safe_platforms')
    known={p.id:p for p in scene.platforms}
    if not isinstance(ids,list) or not ids or any(i not in known or known[i].right-known[i].left<60 for i in ids):
        raise ValueError('Parking requires reviewed, mapped refuge platforms at least 60 pixels wide')
    roi=data.get('health_text_roi')
    if (not isinstance(roi,list) or len(roi)!=4 or any(type(v)!=int for v in roi)
            or not 0<=roi[0]<roi[2]<=scene.width or not 0<=roi[1]<roi[3]<=scene.height):
        raise ValueError('Parking requires a valid health-number observation region')
    return list(dict.fromkeys(ids)),tuple(roi)


def health_signature(image,roi):
    if image is None:return None
    x1,y1,x2,y2=roi;patch=image[y1:y2,x1:x2]
    if patch.size==0:return None
    # Compare the reviewed numeric region, conservatively resetting on healing
    # as well as damage. Approximate bar length is not treated as proof of HP.
    glyphs=(patch.min(axis=2)>180)&(np.ptp(patch.astype(int),axis=2)<45)
    if np.count_nonzero(glyphs)<30 or glyphs.mean()>.6:return None
    return glyphs.tobytes()


class SafeParking:
    def __init__(self,motion,platform_ids,health_roi,quiet_seconds=3):
        self.base=Controller(motion,navigate=True)
        self.platform_ids=tuple(platform_ids);self.health_roi=health_roi
        self.quiet_seconds=quiet_seconds;self.target=None;self.done=False
        self.quiet_since=None;self.last_health=None;self.quiet_position=None
        self.last_observation=None;self.epoch=None;self.last_reason='parking_not_started'
        self.center_settle_until=0

    def reset_confirmation(self):
        self.quiet_since=None;self.last_health=None;self.quiet_position=None

    def result(self):
        return dict(confirmed=self.done,target=self.target,reason=self.last_reason,
            quiet_seconds=self.quiet_seconds if self.done else 0,
            verified_edges=sorted(self.base.verified_edges),failed_edges=sorted(self.base.failures),
            verification='reviewed refuge, stable position, no nearby observed monsters, unchanged numeric HP pixels',
            scope='observed stopping interval; not a guarantee against future monster spawns')

    def decide(self,o,now,image):
        d=self._decide(o,now,image);self.last_reason=d.reason;return d

    def _decide(self,o,now,image):
        if o is None or o.reason or not o.player or not o.motion_valid or not 0<=now-o.captured_at<.085:
            self.reset_confirmation();return Decision(reason='parking_wait_for_localization')
        if self.epoch!=o.map_epoch:
            self.base.reset();self.base.last_epoch=o.map_epoch;self.epoch=o.map_epoch
            self.target=None;self.reset_confirmation()
        if self.last_observation is not None and now-self.last_observation>.2:self.reset_confirmation()
        self.last_observation=now
        # Parking also plans routes before calling the movement controller.
        # Expire temporary edge failures here, including when no route exists.
        self.base.expire_failures(now)
        floor=standing_platform(o);player=o.player.box
        by_id={p.id:p for p in o.platforms}
        enemies=[m for m in [*o.monsters,*o.navigation_targets] if m.confidence>=.78]
        def threatened(p):
            # Live round_037 showed an incoming attack (MISS) on a refuge
            # above and to the side of a monster. A clear platform alone is
            # insufficient; keep a conservative ranged-threat neighbourhood.
            return any(p.left-400<=m.box.cx<=p.right+400 and abs(m.box.y2-p.y)<120 for m in enemies)
        candidates=[by_id[i] for i in self.platform_ids if i in by_id and not threatened(by_id[i])]
        if self.target is not None and self.target not in {p.id for p in candidates}:
            self.target=None;self.base.reset();self.base.last_epoch=o.map_epoch;self.reset_confirmation()
        if self.target is None:
            if floor is None:
                # A fresh parking run may start attached to a rope after the
                # previous session ended. Use the existing bounded attachment
                # probe instead of waiting indefinitely for a ground pose.
                return self.base.decide(replace(o,monsters=[],navigation_targets=[],minimap_goals=[]),now)
            edges=graph(o.platforms,o.ropes,self.base.motion)
            paths=[(p,route(edges,floor.id,p.id,self.base.failures)) for p in candidates]
            paths=[(p,path) for p,path in paths if p.id==floor.id or path]
            if not paths:return Decision(reason='parking_no_reachable_refuge')
            p,_=min(paths,key=lambda item:len(item[1])*3+abs((item[0].left+item[0].right)/2-player.cx)/max(60,self.base.motion.speed))
            self.target=p.id
        target=by_id[self.target]
        # Arrival is confirmed with quiet physical observations, never just a
        # route planner's completion flag or the absence of a detected monster.
        if floor and floor.id==self.target:
            self.base.reset();self.base.last_epoch=o.map_epoch
            midpoint=(floor.left+floor.right)/2
            dx=midpoint-player.cx
            if now<self.center_settle_until:
                self.reset_confirmation();return Decision(reason='parking_center_settle',target=self.target)
            if abs(dx)>12:
                self.reset_confirmation()
                if abs(dx)<=45:self.center_settle_until=now+.35
                return Decision(frozenset({'right' if dx>0 else 'left'}),'parking_center',self.target)
            hp=health_signature(image,self.health_roi)
            position=(player.cx-floor.left,player.y2-floor.y)
            # Pixel jitter of 1–2px produces large per-frame derivatives at
            # 30Hz. Verify bounded position over the whole dwell instead.
            stable=hp is not None
            if self.quiet_position is not None and any(abs(a-b)>5 for a,b in zip(position,self.quiet_position)):stable=False
            if hp is not None and self.last_health is not None and hp!=self.last_health:stable=False
            if not stable:self.reset_confirmation();return Decision(reason='parking_verify_stationary',target=self.target)
            if self.quiet_since is None:
                self.quiet_since=now;self.quiet_position=position
            self.last_health=hp
            if now-self.quiet_since>=self.quiet_seconds:
                self.done=True;return Decision(reason='parking_confirmed',target=self.target)
            return Decision(reason='parking_observe_quiet',target=self.target)
        self.reset_confirmation()
        # Existing movement programs retain collision/rope/failure handling.
        # Remove combat goals so a nearby monster cannot restart the farm loop.
        observation=replace(o,monsters=[],navigation_targets=[],minimap_goals=[])
        if self.base.transition or self.base.rope_climber or self.base.pending_edge:
            return self.base.decide(observation,now)
        if floor is None:return self.base.decide(observation,now)
        edges=graph(o.platforms,o.ropes,self.base.motion)
        path=route(edges,floor.id,self.target,self.base.failures)
        if not path:
            self.target=None;return Decision(reason='parking_route_unavailable')
        self.base.pending_edge=path[0];self.base.pending_since=now
        return self.base._navigate_step(observation,path[0],now)
