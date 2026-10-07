"""Offline-learned platform intent. Predictions are suggestions, never key commands."""
import math
import struct


def observation_row(o,floor):
    def actor(a):return dict(box=list(vars(a.box).values()),confidence=a.confidence,vx=a.vx,vy=a.vy)
    return dict(player=actor(o.player) if o.player else None,reason=o.reason,floor_id=floor.id if floor else None,
                platforms=[vars(p) for p in o.platforms],monsters=[actor(a) for a in o.monsters],
                navigation_targets=[actor(a) for a in o.navigation_targets])


def features(row,platform_ids):
    player=row.get('player')
    if not player or row.get('reason') or not row.get('floor_id'):return None
    b=player['box'];x=(b[0]+b[2])/2;y=b[3]
    floors={p['id']:p for p in row['platforms']}
    current=floors.get(row['floor_id'])
    if not current:return None
    targets=row.get('navigation_targets',[])+row.get('monsters',[])
    distinct=[]
    for m in targets:
        if m['confidence']<.78:continue
        box=m['box'];cx=(box[0]+box[2])/2;foot=box[3]
        if not any(abs(cx-a)<25 and abs(foot-b)<25 for a,b in distinct):distinct.append((cx,foot))
    nearest=min(distinct,key=lambda q:abs(q[0]-x)+abs(q[1]-y)) if distinct else (x,y)
    span=max(1,current['right']-current['left'])
    values=[max(-1,min(2,(x-current['left'])/span)),min(span,1600)/1600,
            player.get('vx',0)/600,player.get('vy',0)/800,
            min(len(distinct),20)/20,(nearest[0]-x)/1366,(nearest[1]-y)/768]
    for ident in platform_ids:
        p=floors.get(ident)
        if p is None:values.extend([0,0,0,0,0]);continue
        count=sum(p['left']<=cx<=p['right'] and abs(foot-p['y'])<45 for cx,foot in distinct)
        visible=(max(0,min(1366,p['right'])-max(0,p['left']))/max(1,p['right']-p['left'])
                 if 0<=p['y']<=694 else 0)
        values.extend([float(ident==row['floor_id']),min(count,8)/8,visible,
                       max(-2,min(2,((p['left']+p['right'])/2-x)/1366)),
                       max(-2,min(2,(p['y']-y)/768))])
    return values if all(math.isfinite(v) for v in values) else None


class PlatformIntent:
    def __init__(self,data):
        self.artifact=data
        if data.get('version')!=1:raise ValueError('Unknown platform policy schema')
        self.platform_ids=data['platform_ids'];self.classes=data['classes'];self.trees=data['trees']
        self.feature_count=7+5*len(self.platform_ids)
        if (not 1<=len(self.platform_ids)<=160 or len(set(self.platform_ids))!=len(self.platform_ids)
                or not set(self.classes)<=set(self.platform_ids) or not 1<=len(self.trees)<=200):
            raise ValueError('Invalid platform policy dimensions')
        for tree in self.trees:
            count=len(tree['left'])
            if not 1<=count<=10000 or any(len(tree[k])!=count for k in ('right','feature','threshold','value')):
                raise ValueError('Invalid platform policy tree')
            for i,(left,right,feat,threshold,value) in enumerate(zip(tree['left'],tree['right'],tree['feature'],tree['threshold'],tree['value'])):
                if len(value)!=len(self.classes) or any(not math.isfinite(v) or v<0 for v in value) or sum(value)<=0:
                    raise ValueError('Invalid policy class weights')
                if not math.isfinite(threshold):raise ValueError('Invalid policy split')
                if left==-1 and right==-1:continue
                if not i<left<count or not i<right<count or not 0<=feat<self.feature_count:
                    raise ValueError('Invalid or cyclic policy tree')

    def predict(self,row):
        x=features(row,self.platform_ids)
        if x is None:return []
        # sklearn evaluates tree inputs as float32. Preserve that boundary in
        # the dependency-free JSON evaluator, including values near a split.
        x=[struct.unpack('f',struct.pack('f',v))[0] for v in x]
        scores=[0.]*len(self.classes)
        for tree in self.trees:
            index=0
            while tree['left'][index]!=-1:
                index=tree['left'][index] if x[tree['feature'][index]]<=tree['threshold'][index] else tree['right'][index]
            value=tree['value'][index];total=sum(value)
            for i,v in enumerate(value):scores[i]+=v/total/len(self.trees)
        return sorted(zip(self.classes,scores),key=lambda q:-q[1])
