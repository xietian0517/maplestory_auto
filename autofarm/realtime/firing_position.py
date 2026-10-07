"""Rank clear intervals on visible platforms, not whole-platform occupancy."""
from .control import graph,route
from .model import Platform


def in_firing_lane(monster,x,direction,target):
    return (target is not None and target.left-12<=monster.box.cx<=target.right+12
            and abs(monster.box.y2-target.y)<40
            and (monster.box.cx>x if direction=='right' else monster.box.cx<x))


def on_combat_platform(monster,platforms):
    return any(p.left-12<=monster.box.cx<=p.right+12 and abs(monster.box.y2-p.y)<40
               for p in platforms)


def clear_intervals(p,monsters,motion,viewport):
    spans=[(max(0,p.left),min(viewport[0],p.right))] if 0<p.y<viewport[1] else []
    clearance=max(85,motion.attack_min+20)
    for m in monsters:
        if abs(m.box.y2-p.y)>=40:continue
        lo,hi=m.box.cx-clearance,m.box.cx+clearance
        spans=[part for a,b in spans for part in ((a,min(b,lo)),(max(a,hi),b)) if part[1]-part[0]>=64]
    return [(a,b) for a,b in spans if b-a>=64]


def choose_position(o,floor,motion,blocked=(),viewport=(1366,694),diagnostics=None,
                    safe_platforms=None, firing_platforms=None, firing_lanes=None, combat_platforms=None):
    monsters=[]
    for m in [*o.monsters,*o.navigation_targets]:
        if m.confidence<.78:continue
        if any(abs(m.box.cx-n.box.cx)<18 and abs(m.box.y2-n.box.y2)<18 for n in monsters):continue
        monsters.append(m)
    windows={p.id:clear_intervals(p,monsters,motion,viewport) for p in o.platforms}
    if safe_platforms is not None:
        # A vacant part of a monster's platform is still a monster platform.
        # Both destinations and intermediate footholds must be reviewed safe.
        for p in o.platforms:
            occupied=any(p.left-20<=m.box.cx<=p.right+20 and abs(m.box.y2-p.y)<40 for m in monsters)
            if p.id not in safe_platforms or occupied:windows[p.id]=[]
    safe={p.id:Platform(p.id,*max(windows[p.id],key=lambda ab:ab[1]-ab[0]),p.y)
          for p in o.platforms if windows[p.id]}
    # Keep the launch interval containing the character, when one is clear.
    source=next(((a,b) for a,b in windows[floor.id] if a<=o.player.box.cx<=b),None)
    safe[floor.id]=Platform(floor.id,*source,floor.y) if source else floor
    choices=[];counts=dict(no_clear_interval=0,no_route=0,no_targets=0);layouts={}
    for p in o.platforms:
        if firing_platforms is not None and p.id not in firing_platforms:continue
        if not windows[p.id]:counts['no_clear_interval']+=1
        for left,right in windows[p.id]:
            dest=Platform(p.id,left,right,p.y);layout=dict(safe);layout[p.id]=dest
            edges=graph(list(layout.values()),o.ropes,motion)
            path=[] if p.id==floor.id else route(edges,floor.id,p.id,blocked)
            if p.id!=floor.id and (not path or len(path)>6):counts['no_route']+=1;continue
            inset=max(18,motion.edge_margin)
            xs=[(left+right)/2,left+inset,right-inset]
            if p.id==floor.id:
                # A same-floor move must not cross an occupied gap.
                if not left<=o.player.box.cx<=right:continue
                xs.append(min(right-inset,max(left+inset,o.player.box.cx)))
            for x in xs:
                targets=[m for m in monsters if -35<m.box.y2-p.y<85
                         and motion.attack_min+20<abs(m.box.cx-x)<motion.attack_max-20]
                if combat_platforms is not None:
                    target_floors=[q for q in o.platforms if q.id in combat_platforms]
                    targets=[m for m in targets if on_combat_platform(m,target_floors)]
                if firing_lanes:
                    lane=firing_lanes.get(p.id)
                    target=next((q for q in o.platforms if lane and q.id==lane['target']),None)
                    targets=[m for m in targets if lane and in_firing_lane(m,x,lane['direction'],target)]
                count=len(targets)
                if not count:counts['no_targets']+=1;continue
                same_floor_risk=any(p.left<=m.box.cx<=p.right and abs(m.box.y2-p.y)<40 for m in monsters)
                score=count*3-len(path)*.5-abs(x-o.player.box.cx)/600-2*same_floor_risk
                item=(score,p.id,x,path,count);choices.append(item)
                layouts[(p.id,x)]={q.id:(q.left,q.right) for q in layout.values()}
    result=max(choices,key=lambda v:(v[0],v[1],-v[2])) if choices else None
    if diagnostics is not None:
        diagnostics.update(rejected=counts,candidates=len(choices),selected=None,bounds={})
        if result:
            diagnostics.update(selected=dict(platform=result[1],x=result[2],targets=result[4],hops=len(result[3])),
                               bounds=layouts[(result[1],result[2])])
    return result


def safe_transfer(o,floor,motion,safe_platforms,firing_platforms,blocked=(),viewport=(1366,694)):
    """Travel between reviewed perches, including toward one below the camera.

    Prefer a visible next landing. A reviewed firing perch just below the HUD
    may receive a short vertical drop with wide overlap; the camera reveals it
    during descent. Monster platforms never enter the graph.
    """
    enemies=[m for m in [*o.monsters,*o.navigation_targets] if m.confidence>=.78]
    platforms=[p for p in o.platforms if p.id in safe_platforms and not any(
        p.left-20<=m.box.cx<=p.right+20 and abs(m.box.y2-p.y)<40 for m in enemies)]
    if floor.id not in {p.id for p in platforms}:platforms.append(floor)
    edges=graph(platforms,o.ropes,motion);by_id={p.id:p for p in platforms};choices=[]
    original_exits=list(edges[floor.id])
    # Filter BEFORE BFS: otherwise a shorter offscreen drop hides an available
    # route through a visible intermediate step.
    edges[floor.id]=[(ident,kind) for ident,kind in edges[floor.id]
        if 20<by_id[ident].y<viewport[1]-15 and by_id[ident].left+30<viewport[0]
        and by_id[ident].right-30>0]
    for ident in firing_platforms:
        if ident==floor.id or ident not in by_id:continue
        path=route(edges,floor.id,ident,blocked)
        if not path:continue
        step=by_id[path[0][1]]
        if not (20<step.y<viewport[1]-15 and step.left+30<viewport[0] and step.right-30>0):continue
        choices.append((len(path),abs(by_id[ident].y-floor.y),ident,path))
    if not choices:
        for ident,kind in original_exits:
            p=by_id[ident]
            if (kind=='drop' and ident in firing_platforms and (floor.id,ident) not in blocked
                and viewport[1]-15<=p.y<viewport[1]+90
                and min(floor.right,p.right)-max(floor.left,p.left)>=64):
                choices.append((1,p.y-floor.y,ident,[(floor.id,ident,'drop')]))
    return min(choices)[3] if choices else []
