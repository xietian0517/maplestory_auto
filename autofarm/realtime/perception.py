"""Fast visual grounding from GPT supplied examples; no map-specific coordinates."""
from dataclasses import dataclass
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor
from copy import copy
import threading
import time
import cv2
import numpy as np

from .model import Actor, Box, Observation, Platform, Rope


@dataclass(frozen=True)
class Frame:
    id: int
    started: float
    finished: float
    image: object
    foreground: bool
    region: tuple
    input_epoch: int = 0


class LatestCapture:
    """Thread-owned MSS instance; one-slot mailbox, never a frame backlog."""
    def __init__(self, api, hwnd, hz=30, epoch_provider=lambda:0):
        self.api=api; self.hwnd=hwnd; self.hz=hz
        self.condition=threading.Condition(); self.done=threading.Event()
        self.latest=None; self.error=None; self.thread=None
        self.backend='initializing'
        self.epoch_provider=epoch_provider

    def _run(self):
        import mss
        try:
            with ExitStack() as stack:
                camera=None
                first=self.api.client_rect(self.hwnd)
                try:
                    import dxcam
                    import ctypes
                    # DXGI output 0 is used only on the primary desktop. Other
                    # displays use MSS until output-origin selection is implemented.
                    sw=ctypes.windll.user32.GetSystemMetrics(0); sh=ctypes.windll.user32.GetSystemMetrics(1)
                    if not (0<=first['left'] and 0<=first['top'] and first['left']+first['width']<=sw
                            and first['top']+first['height']<=sh): raise ValueError('Non-primary output')
                    camera=dxcam.create(output_idx=0,output_color='BGR')
                    stack.callback(camera.release); self.backend='dxgi'
                except (ImportError,OSError,RuntimeError,ValueError):
                    self.backend='mss'
                screen=stack.enter_context(mss.mss()) if camera is None else None
                ident=0; due=time.perf_counter()
                while not self.done.is_set():
                    started=time.perf_counter()
                    input_epoch=self.epoch_provider()
                    region=self.api.client_rect(self.hwnd)
                    fg=self.api.get_foreground()==self.hwnd and not self.api.is_iconic(self.hwnd)
                    if camera:
                        image=camera.grab(region=(region['left'],region['top'],region['left']+region['width'],region['top']+region['height']))
                        if image is None:
                            self.done.wait(.002); continue
                    else:
                        image=np.asarray(screen.grab(region))[:,:,:3].copy()
                    finished=time.perf_counter(); ident+=1
                    # Reject a focus change during capture as well.
                    fg=fg and self.api.get_foreground()==self.hwnd
                    packet=Frame(ident,started,finished,image,fg,tuple(region[k] for k in ('left','top','width','height')),input_epoch)
                    with self.condition:
                        self.latest=packet; self.condition.notify_all()
                    due+=1/self.hz
                    if due<finished: due=finished
                    self.done.wait(max(0,due-time.perf_counter()))
        except Exception as e:
            self.error=type(e).__name__+': '+str(e)
            self.done.set()
            with self.condition: self.condition.notify_all()

    def next(self, previous=0, timeout=.25):
        with self.condition:
            self.condition.wait_for(lambda:self.done.is_set() or self.latest is not None and self.latest.id>previous,timeout)
            return self.latest if self.latest is not None and self.latest.id>previous else None

    def __enter__(self):
        self.thread=threading.Thread(target=self._run,daemon=True); self.thread.start(); return self

    def __exit__(self,*args):
        self.done.set(); self.thread.join(timeout=2)


def crop(image, box):
    return image[round(box.y1):round(box.y2),round(box.x1):round(box.x2)].copy()


def gray_small(image, scale=.5):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)


def iou(a,b):
    area=max(0,min(a.x2,b.x2)-max(a.x1,b.x1))*max(0,min(a.y2,b.y2)-max(a.y1,b.y1))
    return area/max(1,a.width*a.height+b.width*b.height-area)


class Camera:
    """Translation estimate plus aligned appearance check; reset on uncertain views."""
    def __init__(self, seed, area, platforms=(), ropes=(), exclusions=()):
        self.area=area; self.seed=self.prepare(seed); self.previous=self.seed
        self.offset=np.zeros(2); self.bad=0
        self.tick=0
        self.orb=cv2.ORB_create(nfeatures=900,fastThreshold=12,edgeThreshold=12)
        self.current_orb=cv2.ORB_create(nfeatures=1800,fastThreshold=12,edgeThreshold=12)
        scaled=gray_small(seed)
        mask=np.zeros(scaled.shape,np.uint8)
        for p in platforms:
            cv2.rectangle(mask,(round(p.left/2),max(0,round((p.y-8)/2))),
                          (round(p.right/2),round((p.y+28)/2)),255,-1)
        for r in ropes:
            cv2.rectangle(mask,(round((r.x-9)/2),round(r.top/2)),
                          (round((r.x+9)/2),round(r.bottom/2)),255,-1)
        for b in exclusions:
            cv2.rectangle(mask,(max(0,round(b.x1/2)-8),max(0,round(b.y1/2)-8)),
                          (round(b.x2/2)+8,round(b.y2/2)+8),0,-1)
        self.anchor_points,self.anchor_descriptors=self.orb.detectAndCompute(scaled,mask)
        self.anchor_mask=mask
        self.flow_previous=scaled
        self.flow_points=cv2.goodFeaturesToTrack(scaled,200,.01,5,mask=mask)

    def terrain_shift(self,image):
        """Track wood/rope corners between frames instead of the parallax background."""
        current=gray_small(image); shift=None; kept=None
        if self.flow_points is not None and len(self.flow_points)>=12:
            points,status,error=cv2.calcOpticalFlowPyrLK(self.flow_previous,current,self.flow_points,None,
                                                       winSize=(21,21),maxLevel=3)
            if points is not None:
                valid=(status.ravel()==1)&(error.ravel()<20)
                old=self.flow_points.reshape(-1,2)[valid]; new=points.reshape(-1,2)[valid]
                if len(new)>=12:
                    delta=new-old; median=np.median(delta,axis=0)
                    inliers=np.linalg.norm(delta-median,axis=1)<1.5
                    if inliers.sum()>=12 and inliers.mean()>=.5 and max(abs(median))<30:
                        shift=np.median(delta[inliers],axis=0)*2
                        kept=new[inliers].reshape(-1,1,2)
        if kept is None or len(kept)<70:
            offset=self.offset+(shift if shift is not None else 0)
            mask=cv2.warpAffine(self.anchor_mask,np.float32([[1,0,offset[0]/2],[0,1,offset[1]/2]]),
                               (current.shape[1],current.shape[0]))
            kept=cv2.goodFeaturesToTrack(current,200,.01,5,mask=mask)
        self.flow_previous=current; self.flow_points=kept
        return shift

    def anchor(self,image):
        if self.anchor_descriptors is None: return None
        current_mask=cv2.warpAffine(self.anchor_mask,np.float32([[1,0,self.offset[0]/2],[0,1,self.offset[1]/2]]),
                                   (self.anchor_mask.shape[1],self.anchor_mask.shape[0]))
        current_mask=cv2.dilate(current_mask,np.ones((51,51),np.uint8))
        points,descriptors=self.current_orb.detectAndCompute(gray_small(image),current_mask)
        if descriptors is None or len(descriptors)<2: return None
        matches=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(self.anchor_descriptors,descriptors,k=2)
        shifts=[]; origins=[]
        for pair in matches:
            if len(pair)!=2: continue
            a,b=pair
            if a.distance>=.80*b.distance or a.distance>55: continue
            shift=(np.array(points[a.trainIdx].pt)-self.anchor_points[a.queryIdx].pt)*2
            if np.max(np.abs(shift-self.offset))<=50:
                shifts.append(shift); origins.append(self.anchor_points[a.queryIdx].pt)
        if len(shifts)<6: return None
        shifts=np.array(shifts)
        # Translation consensus rejects scrolling backgrounds and repeated logs.
        counts=np.sum(np.linalg.norm(shifts[:,None]-shifts[None,:],axis=2)<5,axis=1)
        center=shifts[np.argmax(counts)]; selected=np.linalg.norm(shifts-center,axis=1)<5
        inliers=shifts[selected]
        if len(inliers)<6 or len(inliers)<len(shifts)*.45: return None
        # Require agreement across terrain features, not one repeated wood log.
        if np.max(np.ptp(np.array(origins)[selected],axis=0))<50: return None
        return np.median(inliers,axis=0)

    def prepare(self,image):
        return gray_small(crop(image,self.area),.25).astype(np.float32)

    def update(self,image):
        current=self.prepare(image)
        if current.shape!=self.previous.shape: return False,(0,0)
        self.tick+=1
        terrain=self.terrain_shift(image)
        if self.tick%4==1:
            anchor=self.anchor(image)
            # Repeated wooden platforms can produce a convincing match one
            # platform-width away. Anchors correct drift, never teleport the map.
            if anchor is not None and np.max(np.abs(anchor-self.offset))<=50:
                self.offset=anchor; self.previous=current; self.bad=0
                return True,tuple(self.offset)
        if terrain is not None:
            self.offset+=terrain; self.previous=current; self.bad=0
            return True,tuple(self.offset)
        shift,response=cv2.phaseCorrelate(self.previous,current)
        sx,sy=shift
        if response<.18 or abs(sx)>35 or abs(sy)>30:
            self.bad+=1; return False,tuple(self.offset)
        warped=cv2.warpAffine(self.previous,np.float32([[1,0,sx],[0,1,sy]]),
                             (current.shape[1],current.shape[0]))
        # Median error tolerates moving sprites but rejects an unrelated map.
        h,w=current.shape
        pad=max(4,int(max(abs(sx),abs(sy)))+2)
        err=float(np.median(np.abs(warped[pad:h-pad,pad:w-pad]-current[pad:h-pad,pad:w-pad])))
        if err>25:
            self.bad+=1; return False,tuple(self.offset)
        self.bad=0
        if abs(sx)>.12: self.offset[0]+=sx*4
        if abs(sy)>.12: self.offset[1]+=sy*4
        self.previous=current
        return True,tuple(self.offset)


class GroundedVision:
    """AI-generated examples are searched anew each frame; tracking never invents hits.

    This is a bootstrapped visual tracker, not a pretrained universal detector.
    New appearances require another semantic keyframe. Small/grayscale matching
    accelerates candidates, then a full-resolution colour score verifies them.
    """
    def __init__(self, scene, seed, name_template=None, async_reacquire=False):
        self.scene=scene; self.seed=seed
        self.camera=Camera(seed,scene.play_area,scene.platforms,scene.ropes,scene.exclusions)
        from .terrain import LocalTerrain
        self.terrain=LocalTerrain(seed,scene.platforms)
        self.name=crop(seed,scene.name_box) if name_template is None else name_template.copy()
        if self.name.shape[:2]!=(round(scene.name_box.height),round(scene.name_box.width)):
            raise ValueError('Identity template must match the semantic name box size')
        self.name_gray=cv2.cvtColor(self.name,cv2.COLOR_BGR2GRAY)
        # White glyphs remain stable when the translucent nameplate background
        # changes. Binary matching also tolerates colour effects behind the text.
        self.name_mask=cv2.inRange(self.name,(181,181,181),(255,255,255))
        self.use_name_mask=np.count_nonzero(self.name_mask)>=20
        if self.name.std()<8: raise ValueError('Player name crop has no texture')
        self.templates=[]
        for b in scene.monsters:
            t=crop(seed,b)
            if t.std()<8: continue
            for im in (t,cv2.flip(t,1)):
                self.templates.append((im,gray_small(im)))
        self.exclusions=[]
        for b in scene.exclusions:
            # Appearance exclusions move with pets/players. Large HUD regions
            # are already outside the combat crop and must not become sprites.
            if b.width<=220 and b.height<=160:
                im=crop(seed,b)
                if im.std()>=8: self.exclusions.append((im,cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)))
        self.previous=[]; self.previous_time=0; self.next_track=1
        self.player_last=None; self.seeded_at=time.perf_counter(); self.offset=(0,0)
        self.player_seen_at=None; self.player_motion=None; self.player_offset=(0,0)
        self.identity_bootstrap=True
        self.partial_stale=False
        cx=round(scene.name_box.cx); foot=round(scene.name_box.cy+scene.foot_offset)
        self.head=seed[max(0,foot-65):max(0,foot-30),max(0,cx-26):min(seed.shape[1],cx+26)].copy()
        from .identity import AppearanceTracker
        self.appearance=AppearanceTracker(seed,cx,foot)
        self.identity_source='none'; self.identity_ticks=0
        self.reacquire_pending=None; self.identity_pending=False
        self.reacquire_executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='identity') if async_reacquire else None
        self.reacquire_future=None; self.reacquire_started=0
        self.target_executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='map_targets') if async_reacquire else None
        self.target_future=None; self.target_started=0; self.target_offset=(0,0)
        self.far_targets=[]; self.far_targets_at=0
        self.recent_monsters=[]

    def close(self):
        if self.target_executor: self.target_executor.shutdown(wait=False,cancel_futures=True)
        if self.reacquire_executor:
            self.reacquire_executor.shutdown(wait=False,cancel_futures=True)

    def _head_confirms(self,image,x,y):
        if self.head.shape[:2]!=(35,52): return False
        foot=round(y+self.scene.foot_offset); x=round(x)
        patch=image[max(0,foot-72):min(image.shape[0],foot-20),max(0,x-35):min(image.shape[1],x+35)]
        if patch.shape[0]<35 or patch.shape[1]<52: return False
        return any(cv2.minMaxLoc(cv2.matchTemplate(patch,t,cv2.TM_CCOEFF_NORMED))[1]>=.8
                   for t in (self.head,cv2.flip(self.head,1)))

    def _player(self,image,dx,dy):
        self.identity_source='none'; self.identity_ticks+=1
        self.identity_pending=False
        if self.reacquire_future is not None and self.reacquire_future.done():
            candidate=self.reacquire_future.result(); self.reacquire_future=None
            if candidate and time.perf_counter()-self.reacquire_started<.35:
                self.reacquire_pending=candidate
        hint=self.player_last
        if self.reacquire_pending:
            hint=(self.reacquire_pending[0],self.reacquire_pending[1]-self.scene.foot_offset)
        if hint is None and self.identity_bootstrap:
            hint=(self.scene.name_box.cx+dx,self.scene.name_box.cy+dy)
        # The sprite is the fast primary cue; periodically confirm the name and
        # learn a bounded set of poses only from a complete name match.
        if hint and self.identity_ticks%10:
            body=self.appearance.locate(image,hint[0],hint[1]+self.scene.foot_offset)
            source='appearance'
            if body is None:
                hx,hy=hint[0],hint[1]+self.scene.foot_offset
                area=(max(0,round(hx-115)),max(0,round(hy-155)),
                      min(image.shape[1],round(hx+115)),min(image.shape[0],round(hy+95)))
                ranked=self.appearance.rank(image,(hx,hy),area)
                if ranked and ranked[2]>=.70 and abs(ranked[0]-hx)<=35 and abs(ranked[1]-hy)<=45:
                    body=ranked; source='appearance_tracked'
            if body:
                x,foot,score=body; self.player_last=(x,foot-self.scene.foot_offset)
                self.identity_source=source
                self.reacquire_pending=None
                return Actor(Box(x-15,foot-48,x+15,foot),score)
        gray=cv2.inRange(image,(181,181,181),(255,255,255)) if self.use_name_mask else cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        template=self.name_mask if self.use_name_mask else self.name_gray
        h,w=template.shape
        area=self.scene.play_area
        if self.player_last:
            x,y=self.player_last
            bounds=Box(max(area.x1,x-100),max(area.y1,y-130),min(area.x2,x+100),min(area.y2,y+100))
        else: bounds=area
        def match(box):
            roi=crop(gray,box)
            if roi.shape[0]<h or roi.shape[1]<w: return None
            scores=cv2.matchTemplate(roi,template,cv2.TM_CCOEFF_NORMED)
            _,score,_,(x,y)=cv2.minMaxLoc(scores)
            if score<.86: return None
            # Reject a second separated name match (e.g. duplicated UI text).
            scores[max(0,y-h//2):y+h//2+1,max(0,x-w//2):x+w//2+1]=0
            if cv2.minMaxLoc(scores)[1]>.86: return None
            return x+round(box.x1)+w/2,y+round(box.y1)+h/2,score
        hit=match(bounds)
        if hit is None and bounds!=area and (not self.reacquire_executor or self.identity_ticks%5==0): hit=match(area)
        full_name=hit is not None
        hint=self.player_last
        if hint is None and self.identity_bootstrap:
            hint=(self.scene.name_box.cx+dx,self.scene.name_box.cy+dy)
        if hit is None and hint and self.use_name_mask:
            # Pet labels can cover either half of the name. Only accept an
            # actually visible, unique text fragment near the previous player.
            lx,ly=hint
            local=Box(max(area.x1,lx-65),max(area.y1,ly-95),min(area.x2,lx+65),min(area.y2,ly+95))
            roi=crop(gray,local); fragments=[]
            for start in (0,w//3,max(0,w-19)):
                end=min(w,start+19); part=template[:,start:end]
                if np.count_nonzero(part)<16: continue
                scores=cv2.matchTemplate(roi,part,cv2.TM_CCOEFF_NORMED)
                _,score,_,(px,py)=cv2.minMaxLoc(scores)
                if score<.92: continue
                scores[max(0,py-h//2):py+h//2+1,max(0,px-10):px+11]=0
                if cv2.minMaxLoc(scores)[1]>.89: continue
                cx=px+round(local.x1)-start+w/2; cy=py+round(local.y1)+h/2
                if abs(cx-lx)<=55 and abs(cy-ly)<=80: fragments.append((cx,cy,score))
            if fragments:
                best=max(fragments,key=lambda f:f[2])
                identity=not self.partial_stale or len(fragments)>=2 or self._head_confirms(image,best[0],best[1])
                if identity and all(abs(f[0]-best[0])<4 and abs(f[1]-best[1])<4 for f in fragments): hit=best
        if hit is None:
            area=self.scene.play_area
            foot_hint=None if hint is None else (hint[0],hint[1]+self.scene.foot_offset)
            if self.reacquire_executor:
                if self.reacquire_future is None:
                    tracker=copy(self.appearance); tracker.poses=list(self.appearance.poses)
                    self.reacquire_started=time.perf_counter()
                    self.reacquire_future=self.reacquire_executor.submit(tracker.rank,image.copy(),foot_hint,
                        tuple(round(v) for v in (area.x1,area.y1,area.x2,area.y2)))
                return None
            candidate=self.appearance.rank(image,foot_hint,tuple(round(v) for v in (area.x1,area.y1,area.x2,area.y2)))
            if candidate is None:
                self.reacquire_pending=None
                return None
            x,foot,score=candidate
            old=self.reacquire_pending
            self.reacquire_pending=candidate
            self.identity_pending=old is None or abs(old[0]-x)>35 or abs(old[1]-foot)>45
            self.identity_source='appearance_candidate' if self.identity_pending else 'appearance_reacquired'
            if not self.identity_pending: self.player_last=(x,foot-self.scene.foot_offset)
            return Actor(Box(x-15,foot-48,x+15,foot),score)
        x,y,score=hit; self.player_last=(x,y)
        foot=y+self.scene.foot_offset
        self.identity_source='name' if full_name else 'name_fragment_and_tracking'
        self.reacquire_pending=None
        if full_name and score>=.93:
            self.appearance.name_updates+=1
            if self.appearance.name_updates%3==0: self.appearance.learn(image,x,foot)
        return Actor(Box(x-15,foot-48,x+15,foot),score)

    def detect_monsters(self,image,roi,player=None):
        combat_image=crop(image,roi); small=gray_small(combat_image); hits=[]
        for original,t in self.templates:
            h,w=t.shape
            if small.shape[0]<h or small.shape[1]<w: continue
            scores=cv2.matchTemplate(small,t,cv2.TM_CCOEFF_NORMED)
            for _ in range(6 if player is None else 4):
                _,score,_,(x,y)=cv2.minMaxLoc(scores)
                if score<.45: break
                scores[max(0,y-h//2):y+h//2+1,max(0,x-w//2):x+w//2+1]=0
                # Refine around the downsampled location at native resolution.
                ox,oy=round(roi.x1)+x*2,round(roi.y1)+y*2
                oh,ow=original.shape[:2]
                rx,ry=max(0,ox-3),max(0,oy-3)
                patch=image[ry:min(image.shape[0],oy+oh+4),rx:min(image.shape[1],ox+ow+4)]
                if patch.shape[0]<oh or patch.shape[1]<ow: continue
                refined=cv2.matchTemplate(patch,original,cv2.TM_CCOEFF_NORMED)
                _,score,_,(tx,ty)=cv2.minMaxLoc(refined)
                if score<.78: continue
                b=Box(rx+tx,ry+ty,rx+tx+ow,ry+ty+oh)
                if player and iou(b,player.box)>.25: continue
                hits.append(Actor(b,float(score)))
        distinct=[]
        for hit in sorted(hits,key=lambda a:a.confidence,reverse=True):
            if not any(iou(hit.box,other.box)>.25 for other in distinct): distinct.append(hit)
        hits=distinct
        kept=[]
        excluded=[]
        if hits:
            # Verify exclusions only around candidate monsters, avoiding a full
            # combat-region correlation for every pet/player example.
            for hit in hits:
                b=hit.box
                for original,t in self.exclusions:
                    h,w=t.shape
                    x1=max(0,round(b.x1-w));y1=max(0,round(b.y1-h))
                    patch=image[y1:min(image.shape[0],round(b.y2+h)),x1:min(image.shape[1],round(b.x2+w))]
                    if patch.shape[0]<h or patch.shape[1]<w: continue
                    scores=cv2.matchTemplate(cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY),t,cv2.TM_CCOEFF_NORMED)
                    _,score,_,(x,y)=cv2.minMaxLoc(scores)
                    if score>=.88: excluded.append(Box(x1+x,y1+y,x1+x+w,y1+y+h))
        for h in sorted(hits,key=lambda x:x.confidence,reverse=True):
            if not any(iou(h.box,b)>.15 for b in excluded) and not any(iou(h.box,k.box)>.25 for k in kept): kept.append(h)
        return kept

    def observe(self,packet,epoch=0):
        image=packet.image
        if image.shape[:2]!=(self.scene.height,self.scene.width):
            return Observation(packet.id,packet.started,reason='window_resized',map_epoch=epoch,motion_valid=False)
        valid,(dx,dy)=self.camera.update(image); self.offset=(dx,dy)
        if not valid:
            self.player_last=None; self.previous=[]; self.player_motion=None
            return Observation(packet.id,packet.started,reason='camera_or_map_changed',map_epoch=epoch,motion_valid=False)
        if self.player_seen_at is not None and packet.started-self.player_seen_at>.5:
            self.partial_stale=True; self.identity_bootstrap=False; self.player_motion=None
        if self.player_last:
            self.player_last=(self.player_last[0]+dx-self.player_offset[0],self.player_last[1]+dy-self.player_offset[1])
        self.player_offset=(dx,dy)
        if self.target_executor:
            if self.target_future is not None and self.target_future.done():
                try: targets=self.target_future.result()
                except cv2.error: targets=[]
                self.target_future=None
                self.far_targets=[Actor(a.box.moved(-self.target_offset[0],-self.target_offset[1]),a.confidence) for a in targets]
                self.far_targets_at=self.target_started
            if self.target_future is None and packet.started-self.target_started>.4:
                self.target_started=packet.started; self.target_offset=(dx,dy)
                self.target_future=self.target_executor.submit(self.detect_monsters,image.copy(),self.scene.play_area)
        navigation_targets=[Actor(a.box.moved(dx,dy),a.confidence) for a in self.far_targets] if packet.started-self.far_targets_at<1.5 else []
        player=self._player(image,dx,dy)
        platforms=[Platform(p.id,p.left+dx,p.right+dx,p.y+dy) for p in self.scene.platforms]
        ropes=[Rope(r.x+dx,r.top+dy,r.bottom+dy) for r in self.scene.ropes]
        if player is None:
            return Observation(packet.id,packet.started,platforms=platforms,ropes=ropes,reason='player_not_found',map_epoch=epoch)
        if self.identity_pending:
            return Observation(packet.id,packet.started,player,platforms=platforms,ropes=ropes,
                               reason='identity_confirming',map_epoch=epoch)
        self.identity_bootstrap=False; self.player_seen_at=packet.started; self.partial_stale=False
        world=(player.box.cx-dx,player.box.y2-dy)
        if self.player_motion:
            px,py,pt=self.player_motion; dt=packet.started-pt
            if .015<=dt<.2:
                player=Actor(player.box,player.confidence,
                             max(-600,min(600,(world[0]-px)/dt)),max(-800,min(800,(world[1]-py)/dt)))
        self.player_motion=(*world,packet.started)
        platforms=self.terrain.update(image,player,platforms,(dx,dy),packet.started)
        area=self.scene.play_area
        # Only presently useful combat/navigation region, not chat or HUD.
        roi=Box(max(area.x1,player.box.cx-430),max(area.y1,player.box.y2-135),
                min(area.x2,player.box.cx+430),min(area.y2,player.box.y2+65))
        kept=self.detect_monsters(image,roi,player)
        full_hits=list(kept)
        from .monster_tracking import recheck
        self.recent_monsters=[r for r in self.recent_monsters if 0<=packet.started-r[0]<=.22]
        for seen,box,sample,old_offset in self.recent_monsters[:6]:
            moved=box.moved(dx-old_offset[0],dy-old_offset[1])
            if abs(moved.cx-player.box.cx)>430 or any(iou(moved,a.box)>.2 for a in kept): continue
            hit=recheck(image,sample,moved)
            if hit and iou(hit.box,player.box)<.25 and not any(iou(hit.box,a.box)>.25 for a in kept): kept.append(hit)
        for hit in full_hits:
            if hit.confidence<.80: continue
            self.recent_monsters=[r for r in self.recent_monsters if iou(r[1].moved(dx-r[3][0],dy-r[3][1]),hit.box)<.25]
            self.recent_monsters.insert(0,(packet.started,hit.box,crop(image,hit.box).copy(),(dx,dy)))
        self.recent_monsters=self.recent_monsters[:6]
        actors=[]; used=set(); dt=packet.started-self.previous_time
        for h in kept:
            candidates=[p for p in self.previous if p.track_id not in used
                        and abs(p.box.cx-h.box.cx)<70 and abs(p.box.cy-h.box.cy)<60]
            prev=min(candidates,key=lambda p:abs(p.box.cx-h.box.cx)+abs(p.box.cy-h.box.cy)) if candidates else None
            if prev and 0<dt<.25:
                vx=max(-450,min(450,(h.box.cx-prev.box.cx)/dt))
                vy=max(-600,min(600,(h.box.cy-prev.box.cy)/dt)); ident=prev.track_id; used.add(ident)
            else:
                vx=vy=0; ident=self.next_track; self.next_track+=1
            actors.append(Actor(h.box,h.confidence,vx,vy,ident))
        self.previous=actors; self.previous_time=packet.started
        return Observation(packet.id,packet.started,player,actors,platforms,ropes,map_epoch=epoch,navigation_targets=navigation_targets)

    def annotate(self,image,o,decision,hz=0,latency=0):
        out=image.copy()
        for p in o.platforms:
            cv2.line(out,(round(p.left),round(p.y)),(round(p.right),round(p.y)),(255,170,40),2)
            cv2.putText(out,p.id,(round(p.left),round(p.y)-4),0,.4,(255,170,40),1)
        for r in o.ropes: cv2.line(out,(round(r.x),round(r.top)),(round(r.x),round(r.bottom)),(255,100,255),2)
        for a,color in [(o.player,(0,255,0)),*((m,(0,230,255)) for m in o.monsters)]:
            if a:
                b=a.box
                cv2.rectangle(out,(round(b.x1),round(b.y1)),(round(b.x2),round(b.y2)),color,2)
        cv2.rectangle(out,(150,0),(min(out.shape[1],1150),50),(20,20,20),-1)
        cv2.putText(out,f'{hz:.1f} decisions/s | frame->decision {latency:.1f}ms | {decision.reason}',(160,22),0,.50,(255,255,255),1)
        cv2.putText(out,f'keys={",".join(sorted(decision.keys)) or "none"} | source=GPT semantic seed + local tracker',(160,43),0,.45,(255,255,255),1)
        return out
