"""Previously observed terrain in the current seed's coordinate frame."""
import json
from pathlib import Path
from .model import Platform,Rope,number


def restore_world(vision,folder):
    calibration=Path(folder)/'minimap_calibration.json'
    if calibration.exists():
        data=json.loads(calibration.read_text(encoding='utf-8'))
        if data.get('request_id')!=vision.scene.request_id:
            raise ValueError('Minimap calibration belongs to another seed')
        vision.minimap.configure(vision.seed,data)
    path=Path(folder)/'world_geometry.json'
    if not path.exists():return
    d=json.loads(path.read_text(encoding='utf-8'))
    if d.get('request_id')!=vision.scene.request_id:raise ValueError('World geometry belongs to another seed')
    ps=d.get('platforms');rs=d.get('ropes')
    if not isinstance(ps,list) or not 1<=len(ps)<=160 or not isinstance(rs,list) or len(rs)>80:
        raise ValueError('Invalid observed world geometry size')
    platforms=[];ids=set()
    for p in ps:
        ident=p['id']
        if not isinstance(ident,str) or not ident or ident in ids:raise ValueError('Invalid world platform ID')
        left,right,y=(number(p[k],-20000,20000) for k in ('left','right','y'))
        if right-left<15:raise ValueError('Invalid world platform span')
        ids.add(ident);platforms.append(Platform(ident,left,right,y))
    if not {p.id for p in vision.scene.platforms}<=ids:raise ValueError('World geometry must retain seed platform IDs')
    ropes=[]
    for r in rs:
        x,top,bottom=(number(r[k],-20000,20000) for k in ('x','top','bottom'))
        if bottom<=top:raise ValueError('Invalid world rope span')
        ropes.append(Rope(x,top,bottom))
    vision.world_platforms=platforms;vision.world_ropes=ropes;vision.world_data=d
