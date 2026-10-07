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
    def __init__(self, api, hwnd, hz=30, epoch_provider=lambda:0, foreground_only=False):
        self.api=api; self.hwnd=hwnd; self.hz=hz
        self.condition=threading.Condition(); self.done=threading.Event()
        self.latest=None; self.error=None; self.thread=None
        self.backend='initializing'
        self.epoch_provider=epoch_provider
        self.foreground_only=foreground_only

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
                    if self.foreground_only and not fg:
                        self.done.wait(.02); due=time.perf_counter(); continue
                    if camera:
                        # A moved or minimized window can leave this output. Do not
                        # crop coordinates or feed a partial frame to grounding.
                        if not (0<=region['left'] and 0<=region['top']
                                and region['left']+region['width']<=sw
                                and region['top']+region['height']<=sh):
                            self.done.wait(.02); due=time.perf_counter(); continue
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
        self.aligned=False
        self.relocalize_at=0
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
        self.relocalization_tiles=[]
        for p in platforms:
            if p.right-p.left<80:continue
            for x in np.linspace(p.left+10,p.right-70,3):
                x=round(x);y=round(p.y)-10
                if x<0 or y<0 or x+64>seed.shape[1] or y+42>seed.shape[0]:continue
                if any(b.x1<x+64 and b.x2>x and b.y1<y+42 and b.y2>y for b in exclusions):continue
                self.relocalization_tiles.append((x,y,seed[y:y+42,x:x+64].copy()))

    def verify_origin(self,image,shifts):
        # ORB ratio matching is sparse on repeated wooden logs. Verify its
        # candidate translations against many separated original colour tiles.
        # A single convincing log/vine cannot establish the map's origin.
        candidates=[]; tested=[]
        for shift in shifts:
            if any(np.linalg.norm(shift-old)<6 for old in tested):continue
            if len(tested)>=16:break
            tested.append(shift);dx,dy=np.rint(shift).astype(int);hits=[];corrections=[];total=0
            for x,y,tile in self.relocalization_tiles:
                xx=x+dx;yy=y+dy
                if xx<3 or yy<3 or xx+67>image.shape[1] or yy+45>self.area.y2:continue
                total+=1
                _,value,_,(tx,ty)=cv2.minMaxLoc(cv2.matchTemplate(image[yy-3:yy+45,xx-3:xx+67],tile,cv2.TM_CCOEFF_NORMED))
                if value>=.80:
                    hits.append((x,y));corrections.append((dx+tx-3,dy+ty-3))
            spread=np.ptp(hits,axis=0) if hits else np.zeros(2)
            distributed=(len(hits)>=8 and len(hits)>=total*.30
                         and min(spread)>=120)
            if len(hits)>=6 and (len(hits)>=total*.45 or distributed):
                spread=np.ptp(hits,axis=0)
                correction=np.median(corrections,axis=0)
                consistent=np.linalg.norm(np.array(corrections)-correction,axis=1)<2
                if max(spread)>=160 and min(spread)>=40 and consistent.mean()>=.8:
                    candidates.append((len(hits),correction))
        candidates.sort(key=lambda c:c[0],reverse=True)
        if not candidates or len(candidates)>1 and candidates[0][0]<candidates[1][0]*1.5:return None
        return candidates[0][1]

    def relocalize(self,image):
        """Align a reused scene before incremental tracking may assume zero drift.

        Require a dominant translation across separated terrain features. Repeated
        log surfaces or one vertical vine alone cannot establish the scene origin.
        """
        if self.anchor_descriptors is None: return None
        orb=cv2.ORB_create(nfeatures=2500,fastThreshold=12,edgeThreshold=12)
        points,descriptors=orb.detectAndCompute(gray_small(image),None)
        if descriptors is None or len(descriptors)<2: return None
        matches=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(self.anchor_descriptors,descriptors,k=2)
        shifts=[]; origins=[]
        for pair in matches:
            if len(pair)!=2: continue
            a,b=pair
            if a.distance>=.75*b.distance or a.distance>50: continue
            shifts.append((np.array(points[a.trainIdx].pt)-self.anchor_points[a.queryIdx].pt)*2)
            origins.append(self.anchor_points[a.queryIdx].pt)
        if len(shifts)<3: return None
        shifts=np.array(shifts)
        votes=np.sum(np.linalg.norm(shifts[:,None]-shifts[None,:],axis=2)<5,axis=1)
        best=int(np.argmax(votes)); keep=np.linalg.norm(shifts-shifts[best],axis=1)<5
        if len(self.relocalization_tiles)>=12:
            return self.verify_origin(image,shifts[np.argsort(votes)[::-1]])
        other=np.linalg.norm(shifts-shifts[best],axis=1)>10
        runner=int(max(votes[other],default=0))
        if keep.sum()<8 or keep.mean()<.3 or keep.sum()<2*runner:
            return self.verify_origin(image,shifts[np.argsort(votes)[::-1]])
        spread=np.ptp(np.array(origins)[keep],axis=0)
        if max(spread)<80 or min(spread)<20: return None
        return np.median(shifts[keep],axis=0)

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
        if not self.aligned:
            # Identical/near-identical seed frames also cover texture-light test
            # scenes. Otherwise establish an absolute origin, never infer it
            # from optical flow between unrelated starting views.
            if self.tick<self.relocalize_at: return False,tuple(self.offset)
            self.relocalize_at=self.tick+12
            if float(np.mean(abs(current-self.seed)))<3:
                origin=np.zeros(2)
            else: origin=self.relocalize(image)
            if origin is None:
                self.bad+=1; return False,tuple(self.offset)
            self.offset=origin; self.aligned=True; self.previous=current
            self.flow_previous=gray_small(image); self.flow_points=None; self.bad=0
            return True,tuple(self.offset)
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
            self.bad+=1
            self.aligned=False; self.relocalize_at=self.tick+1
            return False,tuple(self.offset)
        warped=cv2.warpAffine(self.previous,np.float32([[1,0,sx],[0,1,sy]]),
                             (current.shape[1],current.shape[0]))
        # Median error tolerates moving sprites but rejects an unrelated map.
        h,w=current.shape
        pad=max(4,int(max(abs(sx),abs(sy)))+2)
        err=float(np.median(np.abs(warped[pad:h-pad,pad:w-pad]-current[pad:h-pad,pad:w-pad])))
        if err>25:
            self.bad+=1
            self.aligned=False; self.relocalize_at=self.tick+1
            return False,tuple(self.offset)
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
    def __init__(self, scene, seed, name_template=None, async_reacquire=False, minimap=True):
        self.scene=scene; self.seed=seed
        self.world_platforms=list(scene.platforms);self.world_ropes=list(scene.ropes);self.world_data=None
        from .minimap import MinimapNavigator
        self.minimap=MinimapNavigator(enabled=minimap)
        self.minimap_hint=None
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
        self.previous=[]; self.previous_time=0; self.previous_offset=(0,0); self.next_track=1
        from .monster_tracking import RecentTracks
        self.recent_tracks=RecentTracks()
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
        self.occluded_pending=None
        self.reacquire_executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='identity') if async_reacquire else None
        self.reacquire_future=None; self.reacquire_started=0
        self.target_executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='map_targets') if async_reacquire else None
        self.target_future=None; self.target_started=0; self.target_offset=(0,0)
        self.far_targets=[]; self.far_targets_at=0
        self.recent_monsters=[]
        self.minimap_motion=[]

    def minimap_player(self, now):
        """The yellow marker is the sole live player-position authority."""
        self.identity_source='none'
        hint=self.minimap.player_hint(self.offset,now)
        if hint is None:
            self.minimap_motion=[]
            return None
        x,y=hint
        world=np.array(hint)-self.offset
        self.minimap_motion=[s for s in self.minimap_motion if 0<now-s[0]<=.16]
        vx=vy=0.
        if self.minimap_motion:
            t,previous=self.minimap_motion[0]
            vx,vy=(world-previous)/(now-t)
        self.minimap_motion.append((now,world.copy()))
        self.identity_source='minimap_yellow'
        return Actor(Box(x-15,y-48,x+15,y),.99,float(vx),float(vy))

    def minimap_calibration_data(self):
        if self.minimap.calibration is None:return None
        scale,intercept=self.minimap.calibration
        return dict(request_id=self.scene.request_id,scale=scale,intercept=intercept.tolist())

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
        if hint is None and self.minimap_hint is not None:
            hx,hy=self.minimap_hint
            hint=(hx,hy-self.scene.foot_offset)
        if self.reacquire_pending:
            hint=(self.reacquire_pending[0],self.reacquire_pending[1]-self.scene.foot_offset)
        if hint is None and self.identity_bootstrap:
            hint=(self.scene.name_box.cx+dx,self.scene.name_box.cy+dy)
        # The sprite is the fast primary cue; periodically confirm the name and
        # learn a bounded set of poses only from a complete name match.
        body=None; source='appearance'
        if hint:
            body=self.appearance.locate(image,hint[0],hint[1]+self.scene.foot_offset)
            source='appearance'
            if body is None:
                hx,hy=hint[0],hint[1]+self.scene.foot_offset
                area=(max(0,round(hx-115)),max(0,round(hy-155)),
                      min(image.shape[1],round(hx+115)),min(image.shape[0],round(hy+95)))
                ranked=self.appearance.rank(image,(hx,hy),area)
                if ranked and ranked[2]>=.70 and abs(ranked[0]-hx)<=35 and abs(ranked[1]-hy)<=45:
                    body=ranked; source='appearance_tracked'
            if body and self.identity_ticks%10:
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
                if roi.shape[0]<h or roi.shape[1]<part.shape[1]: continue
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
            # A scheduled name check must not manufacture a lost frame when a
            # pet obscures the label but both head and clothing still match.
            if body:
                x,foot,score=body; self.player_last=(x,foot-self.scene.foot_offset)
                self.identity_source=source
                self.reacquire_pending=None
                return Actor(Box(x-15,foot-48,x+15,foot),score)
            area=self.scene.play_area
            head_area=tuple(round(v) for v in (area.x1,area.y1,area.x2,area.y2))
            head_hint=self.occluded_pending
            if head_hint is None and self.player_last:
                head_hint=(self.player_last[0],self.player_last[1]+self.scene.foot_offset)
            if head_hint:
                hx,hy=head_hint[:2]
                head_area=(max(0,round(hx-90)),max(0,round(hy-145)),
                           min(image.shape[1],round(hx+90)),min(round(area.y2),round(hy+65)))
            head=self.appearance.occluded_head(image,head_area)
            if head is None and head_hint and self.identity_ticks%10==0:
                head=self.appearance.occluded_head(image,tuple(round(v) for v in (area.x1,area.y1,area.x2,area.y2)))
            if head:
                hx,hy=head[:2]
                body_area=(max(0,round(hx-45)),max(0,round(hy-80)),
                           min(image.shape[1],round(hx+45)),min(image.shape[0],round(hy+15)))
                if self.appearance.rank(image,(hx,hy),body_area) is not None:
                    head=None  # Preserve the stronger full-body reacquisition path.
            old=self.occluded_pending; self.occluded_pending=head
            if head:
                x,foot,score=head
                self.identity_pending=old is None or abs(old[0]-x)>20 or abs(old[1]-foot)>25
                self.identity_source='head_face_candidate' if self.identity_pending else 'head_face_occluded'
                if not self.identity_pending:self.player_last=(x,foot-self.scene.foot_offset)
                return Actor(Box(x-15,foot-48,x+15,foot),score)
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

    def combat_roi(self,player):
        """Include complete sprites whose feet are in the nearby firing band."""
        area=self.scene.play_area
        height=max((im.shape[0] for im,_ in self.templates),default=0)
        half_width=max((im.shape[1]/2 for im,_ in self.templates),default=0)
        # A fixed 135px headroom cut the ~150px golem's head off even on
        # the same floor. Match full sprites, including slightly higher foes.
        above=max(135,height+80)
        reach=430+half_width
        return Box(max(area.x1,player.box.cx-reach),max(area.y1,player.box.y2-above),
                   min(area.x2,player.box.cx+reach),min(area.y2,player.box.y2+110))

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
        hits.extend(self._tall_monsters(image,roi,small,player))
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

    def _tall_monsters(self,image,roi,small,player):
        """Verify stable head/core parts when tall sprites animate their limbs.

        The full-body and brightness checks still require a visible matching
        creature. A head alone, a dark scenery statue, or a cached track cannot
        become an attack target through this fallback.
        """
        if getattr(self,'_tall_template_count',None)!=len(self.templates):
            parts=[]
            for original,_ in self.templates:
                h,w=original.shape[:2]
                if h<100 or w<90:continue
                x1,y1,x2,y2=round(w*.25),round(h*.10),round(w*.75),round(h*.44)
                head=original[y1:y2,x1:x2]
                core=(slice(round(h*.12),round(h*.68)),slice(round(w*.28),round(w*.72)))
                parts.append((original,head,gray_small(head),(x1,y1,x2,y2),core))
            self._tall_parts=parts;self._tall_template_count=len(self.templates)
        hits=[]
        for original,head,t,(x1,y1,x2,y2),core in self._tall_parts:
            h,w=t.shape
            if small.shape[0]<h or small.shape[1]<w:continue
            scores=cv2.matchTemplate(small,t,cv2.TM_CCOEFF_NORMED)
            for _ in range(4):
                _,score,_,(x,y)=cv2.minMaxLoc(scores)
                if score<.65:break
                scores[max(0,y-h//2):y+h//2+1,max(0,x-w//2):x+w//2+1]=0
                ox,oy=round(roi.x1)+x*2,round(roi.y1)+y*2
                rx,ry=max(0,ox-3),max(0,oy-3)
                hh,hw=head.shape[:2]
                patch=image[ry:min(image.shape[0],oy+hh+4),rx:min(image.shape[1],ox+hw+4)]
                if patch.shape[0]<hh or patch.shape[1]<hw:continue
                refined=cv2.matchTemplate(patch,head,cv2.TM_CCOEFF_NORMED)
                _,head_score,_,(tx,ty)=cv2.minMaxLoc(refined)
                if head_score<.90:continue
                bx,by=rx+tx-x1,ry+ty-y1
                oh,ow=original.shape[:2]
                if bx<roi.x1 or by<roi.y1 or bx+ow>roi.x2 or by+oh>roi.y2:continue
                body=image[by:by+oh,bx:bx+ow]
                # Correlation alone is invariant to the deep doorway shadow.
                mean_delta=abs(body[y1:y2,x1:x2].mean(axis=(0,1))-head.mean(axis=(0,1)))
                if max(mean_delta)>30:continue
                core_score=float(cv2.matchTemplate(body[core],original[core],cv2.TM_CCOEFF_NORMED)[0,0])
                if core_score<.78:continue
                body_score=float(cv2.matchTemplate(body,original,cv2.TM_CCOEFF_NORMED)[0,0])
                if body_score<.55:continue
                b=Box(bx,by,bx+ow,by+oh)
                if player and iou(b,player.box)>.25:continue
                hits.append(Actor(b,min(float(head_score),core_score)))
        return hits

    def observe(self,packet,epoch=0):
        image=packet.image
        if image.shape[:2]!=(self.scene.height,self.scene.width):
            return Observation(packet.id,packet.started,reason='window_resized',map_epoch=epoch,motion_valid=False)
        self.minimap.observe(image,packet.started)
        valid,(dx,dy)=self.camera.update(image); self.offset=(dx,dy)
        self.minimap_hint=self.minimap.player_hint((dx,dy),packet.started) if valid else None
        if not valid:
            self.player_last=None; self.previous=[]; self.player_motion=None
            self.recent_tracks.clear()
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
        player=self.minimap_player(packet.started)
        platforms=[Platform(p.id,p.left+dx,p.right+dx,p.y+dy) for p in self.world_platforms]
        ropes=[Rope(r.x+dx,r.top+dy,r.bottom+dy) for r in self.world_ropes]
        if player is None:
            reason=('minimap_uncalibrated' if self.minimap.calibration is None
                    else 'minimap_'+self.minimap.reason)
            return Observation(packet.id,packet.started,platforms=platforms,ropes=ropes,reason=reason,map_epoch=epoch)
        if self.identity_pending:
            return Observation(packet.id,packet.started,player,platforms=platforms,ropes=ropes,
                               reason='identity_confirming',map_epoch=epoch)
        self.identity_bootstrap=False; self.player_seen_at=packet.started; self.partial_stale=False
        world=(player.box.cx-dx,player.box.y2-dy)
        if self.player_motion and self.identity_source!='minimap_yellow':
            px,py,pt=self.player_motion; dt=packet.started-pt
            if .015<=dt<.2:
                player=Actor(player.box,player.confidence,
                             max(-600,min(600,(world[0]-px)/dt)),max(-800,min(800,(world[1]-py)/dt)))
        self.player_motion=(*world,packet.started)
        platforms=self.terrain.update(image,player,platforms,(dx,dy),packet.started)
        minimap_goals=self.minimap.goals(platforms,player,(dx,dy),packet.started,self.scene.play_area)
        area=self.scene.play_area
        # Only presently useful combat/navigation region, not chat or HUD.
        roi=self.combat_roi(player)
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
        from .monster_tracking import associate
        new_id_start=self.next_track
        actors,self.next_track=associate(kept,self.previous,packet.started-self.previous_time,
            (dx-self.previous_offset[0],dy-self.previous_offset[1]),self.next_track)
        actors=self.recent_tracks.update(actors,(dx,dy),packet.started,new_id_start)
        self.previous=actors; self.previous_time=packet.started; self.previous_offset=(dx,dy)
        return Observation(packet.id,packet.started,player,actors,platforms,ropes,map_epoch=epoch,
                           navigation_targets=navigation_targets,minimap_goals=minimap_goals,
                           position_quantum=1/self.minimap.scale)

    def annotate(self,image,o,decision,hz=0,latency=0):
        out=image.copy()
        self.minimap.annotate(out)
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
