"""Confirm local support from platform appearance instead of a complete map."""
import cv2
import numpy as np
from .model import Platform


class LocalTerrain:
    def __init__(self,seed,platforms):
        self.tiles=[]; self.extensions={}; self.last_scan=-1
        for p in sorted(platforms,key=lambda q:q.right-q.left,reverse=True)[:2]:
            y=round(p.y)+4
            for x in np.linspace(p.left+8,p.right-32,6):
                x=round(x); tile=seed[y:y+18,x:x+24]
                if tile.shape==(18,24,3) and tile.std()>15:
                    self.tiles.append((tile.copy(),cv2.cvtColor(tile,cv2.COLOR_BGR2GRAY)))

    def support(self,image,cx,foot):
        x1=max(0,round(cx-240)); x2=min(image.shape[1],round(cx+240))
        y1=max(0,round(foot-10)); y2=min(image.shape[0],round(foot+38))
        roi=image[y1:y2,x1:x2]
        if roi.shape[0]<18 or roi.shape[1]<24: return None
        gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY); hits=[]
        for tile,t in self.tiles:
            scores=cv2.matchTemplate(gray,t,cv2.TM_CCOEFF_NORMED)
            for _ in range(24):
                _,v,_,(x,y)=cv2.minMaxLoc(scores)
                if v<.68: break
                scores[max(0,y-3):y+4,max(0,x-7):x+8]=0
                if cv2.matchTemplate(roi[y:y+18,x:x+24],tile,cv2.TM_CCOEFF_NORMED)[0,0]<.68: continue
                hits.append((x1+x,x1+x+24,y1+y-4))
        candidates=[]
        for row in {round(h[2]/4)*4 for h in hits}:
            spans=sorted((a,b) for a,b,y in hits if abs(y-row)<=4)
            groups=[]
            for a,b in spans:
                if groups and a-groups[-1][1]<=10: groups[-1][1]=max(b,groups[-1][1])
                else: groups.append([a,b])
            for a,b in groups:
                if b-a>=48 and a<=cx<=b and abs(row-foot)<=12:
                    candidates.append((a,b,row))
        return min(candidates,key=lambda p:abs(p[2]-foot)) if candidates else None

    def update(self,image,player,platforms,offset,now):
        dx,dy=offset
        current={p.id:p for p in platforms}
        for key,p in self.extensions.items():
            current[key]=Platform(key,p.left+dx,p.right+dx,p.y+dy)
        if abs(player.vy)<80 and now-self.last_scan>=.16:
            near=[p for p in current.values() if p.left<=player.box.cx<=p.right and abs(p.y-player.box.y2)<16]
            if not near or min(player.box.cx-near[0].left,near[0].right-player.box.cx)<80:
                self.last_scan=now
                found=self.support(image,player.box.cx,player.box.y2)
                if found:
                    a,b,y=found
                    overlaps=[p for p in current.values() if abs(p.y-y)<10 and min(p.right,b)>max(p.left,a)]
                    if overlaps:
                        p=min(overlaps,key=lambda p:abs(p.y-y)); key=p.id
                        a=min(a,p.left); b=max(b,p.right); y=p.y
                    else: key='local_'+str(len(self.extensions))
                    updated=Platform(key,a,b,y);current[key]=updated
                    self.extensions[key]=Platform(key,a-dx,b-dx,y-dy)
        return list(current.values())
