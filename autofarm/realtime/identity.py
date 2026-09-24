"""Local appearance tracking with name-confirmed pose learning."""
import cv2
import numpy as np


class AppearanceTracker:
    def __init__(self,seed,cx,foot):
        self.poses=[]; self.name_updates=0
        self.learn(seed,cx,foot)

    def learn(self,image,cx,foot):
        x,y=round(cx),round(foot)
        if x<26 or x+26>image.shape[1] or y<65 or y>image.shape[0]: return
        sample=image[y-65:y,x-26:x+26].copy()
        if sample.std()<15: return
        if any(cv2.matchTemplate(sample,p,cv2.TM_CCOEFF_NORMED)[0,0]>.92 for p in self.poses): return
        self.poses.append(sample)
        if len(self.poses)>3: self.poses.pop(1)  # Keep the original identity seed.

    def locate(self,image,cx,foot):
        x1=max(0,round(cx-85)); x2=min(image.shape[1],round(cx+85))
        y1=max(0,round(foot-150)); y2=min(image.shape[0],round(foot+95))
        roi=image[y1:y2,x1:x2]
        if roi.shape[0]<65 or roi.shape[1]<52: return None
        small=cv2.resize(cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY),None,fx=.5,fy=.5)
        candidates=[]
        for pose in self.poses:
            for template in (pose,cv2.flip(pose,1)):
                reduced=cv2.resize(cv2.cvtColor(template,cv2.COLOR_BGR2GRAY),None,fx=.5,fy=.5)
                scores=cv2.matchTemplate(small,reduced,cv2.TM_CCOEFF_NORMED)
                _,score,_,(x,y)=cv2.minMaxLoc(scores)
                if score<.78: continue
                scores[max(0,y-15):y+16,max(0,x-10):x+11]=0
                if cv2.minMaxLoc(scores)[1]>.84: continue
                rx=max(0,x*2-3);ry=max(0,y*2-3)
                patch=roi[ry:min(roi.shape[0],y*2+69),rx:min(roi.shape[1],x*2+56)]
                if patch.shape[0]<65 or patch.shape[1]<52: continue
                refined=cv2.matchTemplate(patch,template,cv2.TM_CCOEFF_NORMED)
                _,score,_,(tx,ty)=cv2.minMaxLoc(refined);x=rx+tx;y=ry+ty
                if score<.87: continue
                # A head alone is insufficient: the clothing/torso must agree.
                body=roi[y+34:y+65,x:x+52]
                body_score=float(cv2.matchTemplate(body,template[34:],cv2.TM_CCOEFF_NORMED)[0,0])
                if body_score<.75: continue
                candidates.append((x1+x+26,y1+y+65,float(score)))
        if not candidates: return None
        best=max(candidates,key=lambda p:p[2])
        if any(abs(p[0]-best[0])>12 or abs(p[1]-best[1])>12 for p in candidates): return None
        return best

    def rank(self,image,hint=None,area=None):
        """Reacquire from visible head AND clothing, including outside the old ROI.

        Returns the best distinctive candidate, not an unconditional argmax on
        empty scenery. Spatial continuity is a small tie-breaker, never identity.
        """
        x1,y1,x2,y2=(0,0,image.shape[1],image.shape[0]) if area is None else area
        roi=image[y1:y2,x1:x2]
        if roi.shape[0]<65 or roi.shape[1]<52: return None
        gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
        candidates=[]
        poses=[self.poses[0],self.poses[-1]] if len(self.poses)>2 and roi.shape[1]>400 else self.poses
        for pose in poses:
            for template in (pose,cv2.flip(pose,1)):
                head=template[:35,6:46]
                scores=cv2.matchTemplate(gray,cv2.cvtColor(head,cv2.COLOR_BGR2GRAY),cv2.TM_CCOEFF_NORMED)
                for _ in range(3):
                    _,coarse,_,(x,y)=cv2.minMaxLoc(scores)
                    if coarse<.55: break
                    scores[max(0,y-24):y+25,max(0,x-24):x+25]=-1
                    rx=max(0,x-3); ry=max(0,y-3)
                    patch=roi[ry:min(roi.shape[0],y+39),rx:min(roi.shape[1],x+44)]
                    if patch.shape[0]<35 or patch.shape[1]<40: continue
                    _,hs,_,(tx,ty)=cv2.minMaxLoc(cv2.matchTemplate(patch,head,cv2.TM_CCOEFF_NORMED))
                    px=rx+tx-6; py=ry+ty
                    if px<0: continue
                    body=roi[py:py+65,px:px+52]
                    if body.shape!=template.shape: continue
                    # Compare the sprite interior. The wide template includes
                    # scenery on both sides, which changes with parallax or
                    # vines even while the same character remains visible.
                    hs=float(cv2.matchTemplate(body[:35,6:46],template[:35,6:46],cv2.TM_CCOEFF_NORMED)[0,0])
                    ts=float(cv2.matchTemplate(body[34:,14:38],template[34:,14:38],cv2.TM_CCOEFF_NORMED)[0,0])
                    score=.55*hs+.45*ts
                    if hs<.65 or ts<.45 or score<.68: continue
                    cx=x1+px+26; foot=y1+py+65
                    continuity=0 if hint is None else .04*max(0,1-np.hypot(cx-hint[0],foot-hint[1])/200)
                    candidate=(cx,foot,score,score+continuity)
                    near=next((i for i,p in enumerate(candidates) if abs(p[0]-cx)<15 and abs(p[1]-foot)<15),None)
                    if near is None: candidates.append(candidate)
                    elif candidates[near][3]<candidate[3]: candidates[near]=candidate
        candidates.sort(key=lambda p:p[3],reverse=True)
        if not candidates: return None
        if len(candidates)>1 and candidates[0][3]-candidates[1][3]<.06: return None
        return candidates[0][:3]
