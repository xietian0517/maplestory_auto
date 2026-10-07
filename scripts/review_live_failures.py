"""Link prolonged controller failures to exact recorded frames, without replay inputs."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import cv2


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def review(folder):
    folder=Path(folder); out=folder/'failure_review'; out.mkdir(exist_ok=True)
    decisions=[r for r in rows(folder/'frames.jsonl') if r.get('phase')=='farming']
    videos={r['frame_id']:r for r in rows(folder/'evaluation/frames.jsonl')}
    totals=defaultdict(float); spans=[]; current=None
    for i,row in enumerate(decisions):
        # Attribute only the observed interval until the next decision. A gap
        # over 200 ms is unobserved, not proof that the prior state persisted.
        duration=decisions[i+1]['t']-row['t'] if i+1<len(decisions) else 0
        duration=duration if 0<=duration<=.2 else 0
        reason=row['reason'];totals[reason]+=duration
        if current is None or current['reason']!=reason or duration==0:
            current=dict(reason=reason,start=row['t'],end=row['t'],seconds=0,frame_ids=[])
            spans.append(current)
        current['end']=row['t']+duration;current['seconds']+=duration
        current['frame_ids'].append(row['frame_id'])
    failures={'player_not_found','identity_confirming','airborne_or_floor_unknown',
              'jump_attack_target_occluded','climb_wait_for_floor','rope_align','rope_approach'}
    chosen=sorted((s for s in spans if s['reason'] in failures and s['seconds']>=.5),
                  key=lambda s:s['seconds'],reverse=True)[:20]
    evidence=[]
    for index,span in enumerate(chosen):
        frames=[]; ids=span['frame_ids']
        for label,ident in [('first',ids[0]),('middle',ids[len(ids)//2]),('last',ids[-1])]:
            v=videos.get(ident)
            if v is None:
                frames.append(dict(frame_id=ident,missing_video=True));continue
            capture=cv2.VideoCapture(str(folder/'evaluation'/v['segment']))
            capture.set(cv2.CAP_PROP_POS_FRAMES,v['video_frame']);ok,image=capture.read();capture.release()
            filename=f'{index:02d}_{label}.png'
            if ok:cv2.imwrite(str(out/filename),image)
            frames.append(dict(frame_id=ident,video_time=v['capture_started'],
                segment=v['segment'],video_frame=v['video_frame'],image=filename if ok else None))
        evidence.append({k:v for k,v in span.items() if k!='frame_ids'}|dict(frames=frames))
    result=dict(reason_seconds=dict(sorted(totals.items(),key=lambda p:-p[1])),episodes=evidence,
        measurement='Decision intervals <= 200 ms; exact frame-ID joins to video clock; no inferred damage or kills.')
    (out/'diagnostics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 实测失败录像索引','','仅统计刷怪阶段；使用帧编号关联录像，不混用两套时钟。','',
           '| 状态 | 累计秒数 |','| --- | ---: |']
    lines += [f'| {k} | {v:.2f} |' for k,v in result['reason_seconds'].items()]
    for e in evidence:
        lines += ['',f"## {e['reason']} · 连续 {e['seconds']:.2f} 秒",'']
        for f in e['frames']:
            if f.get('image'):lines += [f"录像 {f['video_time']:.3f}s，帧 {f['frame_id']}",f"![录像帧]({f['image']})",'']
    (out/'review.md').write_text('\n'.join(lines),encoding='utf-8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder');args=parser.parse_args()
    result=review(args.folder)
    print(json.dumps(dict(reason_seconds=result['reason_seconds'],episodes=len(result['episodes'])),indent=2))
