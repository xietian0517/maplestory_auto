"""Brief local sprite confirmation when a monster changes its animation pose."""
import cv2
import numpy as np
from .model import Actor,Box


def associate(hits,previous,dt,camera_delta,next_track):
    """Associate visible sprites and measure motion relative to the terrain."""
    dx,dy=camera_delta
    previous=[Actor(p.box.moved(dx,dy),p.confidence,p.vx,p.vy,p.track_id)
              for p in previous] if 0<dt<.25 else []
    actors=[];used=set()
    for hit in hits:
        candidates=[p for p in previous if p.track_id not in used
                    and abs(p.box.cx-hit.box.cx)<70 and abs(p.box.cy-hit.box.cy)<60]
        old=min(candidates,key=lambda p:abs(p.box.cx-hit.box.cx)+abs(p.box.cy-hit.box.cy)) if candidates else None
        if old:
            vx=max(-450,min(450,(hit.box.cx-old.box.cx)/dt))
            vy=max(-600,min(600,(hit.box.cy-old.box.cy)/dt))
            ident=old.track_id;used.add(ident)
        else:
            vx=vy=0;ident=next_track;next_track+=1
        actors.append(Actor(hit.box,hit.confidence,vx,vy,ident))
    return actors,next_track


class RecentTracks:
    """Reconnect only current detections across brief, unambiguous gaps.

    Cached boxes never become observations or attack targets. World-space
    matching removes camera translation; ambiguity retains a fresh identity.
    """
    def __init__(self):self.seen={}

    def clear(self):self.seen.clear()

    def update(self,actors,offset,now,new_id_start):
        dx,dy=offset
        self.seen={i:v for i,v in self.seen.items() if 0<now-v[0]<=.22}
        occupied={a.track_id for a in actors};options={}
        for index,a in enumerate(actors):
            if a.track_id<new_id_start:continue
            x,y=a.box.cx-dx,a.box.y2-dy
            options[index]=[i for i,(t,b) in self.seen.items() if i not in occupied
                            and abs(x-b.cx)<35 and abs(y-b.y2)<25]
        proposed={index:ids[0] for index,ids in options.items() if len(ids)==1}
        result=[]
        for index,a in enumerate(actors):
            old_id=proposed.get(index)
            if old_id is not None and sum(old_id in ids for ids in options.values())==1:
                t,b=self.seen[old_id];dt=now-t
                a=Actor(a.box,a.confidence,max(-450,min(450,(a.box.cx-dx-b.cx)/dt)),
                        max(-600,min(600,(a.box.y2-dy-b.y2)/dt)),old_id)
            result.append(a)
        for a in result:self.seen[a.track_id]=(now,a.box.moved(-dx,-dy))
        return result


def color_hist(image):
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    h=cv2.calcHist([hsv],[0,1],None,[24,12],[0,180,0,256])
    return cv2.normalize(h,h).flatten()


def recheck(image,sample,box,dx=0,dy=0):
    h,w=sample.shape[:2]
    if min(h,w)<20: return None
    # Legs/body may stay visible while a face changes to a hit pose. Require
    # both strong current texture and matching upper-body colour distribution.
    left=max(1,round(w*.1));right=min(w-1,round(w*.9));split=round(h*.6)
    part=sample[split:,left:right]
    origin_x=round(box.x1+dx);origin_y=round(box.y1+dy)
    x1=max(0,origin_x-30+left);y1=max(0,origin_y-20+split)
    x2=min(image.shape[1],origin_x+30+right);y2=min(image.shape[0],origin_y+20+h)
    roi=image[y1:y2,x1:x2]
    if roi.shape[0]<part.shape[0] or roi.shape[1]<part.shape[1]: return None
    scores=cv2.matchTemplate(roi,part,cv2.TM_CCOEFF_NORMED)
    _,score,_,(x,y)=cv2.minMaxLoc(scores)
    if score<.90: return None
    x=x1+x-left;y=y1+y-split
    if x<0 or y<0 or x+w>image.shape[1] or y+h>image.shape[0]: return None
    upper=image[y:y+split,x+left:x+right]
    distance=cv2.compareHist(color_hist(sample[:split,left:right]),color_hist(upper),cv2.HISTCMP_BHATTACHARYYA)
    if distance>.50: return None
    return Actor(Box(x,y,x+w,y+h),min(.9,.78+(score-.9)),0,0)
