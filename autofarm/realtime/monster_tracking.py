"""Brief local sprite confirmation when a monster changes its animation pose."""
import cv2
import numpy as np
from .model import Actor,Box


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
