"""Merge inspected screenshots only after verified terrain registration."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from autofarm.realtime.semantic import load_scene,atomic_json
from autofarm.realtime.perception import Camera


def merge_rope(ropes,r,translation,play_area,conflicts):
    dx,dy=translation
    item=dict(x=r.x-dx,top=r.top-dy,bottom=r.bottom-dy,
              top_clipped=r.top<=play_area.y1+5,bottom_clipped=r.bottom>=play_area.y2-5)
    same=[q for q in ropes if abs(q['x']-item['x'])<=20
          and min(q['bottom'],item['bottom'])>max(q['top'],item['top'])
          and (abs(q['top']-item['top'])<=20 or q['top_clipped'] or item['top_clipped'])]
    if not same:ropes.append(item);return
    q=min(same,key=lambda q:abs(q['x']-item['x']))
    if abs(q['x']-item['x'])>7:
        conflicts.append(dict(retained=q.copy(),ignored=item,reason='rope_x_disagreement'));return
    # New observations may extend a viewport-clipped endpoint, but must not
    # stretch a fully observed rope just because an older annotation is longer.
    for key,clipped,extends in [('top','top_clipped',item['top']<q['top']),
                                 ('bottom','bottom_clipped',item['bottom']>q['bottom'])]:
        if q[clipped] and extends:
            q[key]=item[key];q[clipped]=item[clipped]
        elif abs(q[key]-item[key])>15 and not item[clipped]:
            conflicts.append(dict(retained=q.copy(),ignored=item.copy(),reason='rope_'+key+'_disagreement'))


def build(base_folder,source_folders):
    scene,seed=load_scene(base_folder)
    platforms=[vars(p).copy() for p in scene.platforms]
    ropes=[dict(**vars(r),top_clipped=r.top<=scene.play_area.y1+5,
                bottom_clipped=r.bottom>=scene.play_area.y2-5) for r in scene.ropes]
    conflicts=[]
    anchors=[(Path(base_folder),scene,seed,np.zeros(2))];sources=[]
    for folder in source_folders:
        other,image=load_scene(folder);translation=None;via=None
        if (other.width,other.height)!=(scene.width,scene.height):raise ValueError('Different capture scale')
        for anchor_folder,a,im,offset in anchors:
            c=Camera(im,a.play_area,a.platforms,a.ropes,a.exclusions)
            relative=c.relocalize(image)
            if relative is not None:translation=offset+relative;via=a.request_id;break
        if translation is None:raise ValueError('Cannot verify terrain registration: '+str(folder))
        dx,dy=translation
        for p in other.platforms:
            left,right,y=p.left-dx,p.right-dx,p.y-dy
            same=[q for q in platforms if abs(q['y']-y)<=8 and min(q['right'],right)-max(q['left'],left)>15]
            if same:
                q=min(same,key=lambda q:abs(q['y']-y))
                q['left']=min(q['left'],left);q['right']=max(q['right'],right)
            else:platforms.append(dict(id='observed_'+str(len(platforms)),left=left,right=right,y=y))
        for r in other.ropes:
            merge_rope(ropes,r,translation,other.play_area,conflicts)
        request=json.loads((Path(folder)/'request.json').read_text(encoding='utf-8'))
        sources.append(dict(request_id=other.request_id,image_sha256=request['image_sha256'],
            registered_via=via,seed_to_source_translation=list(translation),source_folder=str(Path(folder).resolve())))
        anchors.append((Path(folder),other,image,translation))
    data=dict(request_id=scene.request_id,platforms=platforms,
        ropes=[{k:r[k] for k in ('x','top','bottom')} for r in ropes],sources=sources,
        unresolved_rope_observations=conflicts,
        provenance='Previously inspected scene annotations; translations verified against separated terrain colour patches. Not generated from minimap hints.')
    atomic_json(Path(base_folder)/'world_geometry.json',data)
    return data


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('base');p.add_argument('sources',nargs='+');a=p.parse_args()
    d=build(a.base,a.sources);print(json.dumps(dict(platforms=len(d['platforms']),ropes=len(d['ropes']),sources=d['sources']),ensure_ascii=False,indent=2))
