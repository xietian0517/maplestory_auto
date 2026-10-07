"""Offline recording quality checks and input intervals; no game interaction."""
from collections import Counter
from itertools import groupby
import json
from pathlib import Path

import cv2

from .semantic import atomic_json


def read_rows(path):
    rows=[];truncated=0
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            try:rows.append(json.loads(line))
            except ValueError:truncated+=1
    return rows,truncated


def tags(keys):
    keys=set(keys);result=[]
    if 'shift' in keys:result.append('attack_input')
    if {'alt','down'}<=keys:result.append('drop_attempt_input')
    elif {'alt','up'}<=keys:result.append('rope_catch_attempt_input')
    elif 'alt' in keys:result.append('jump_input')
    if 'up' in keys:result.append('up_input')
    if 'down' in keys:result.append('down_input')
    if keys&{'left','right'}:result.append('horizontal_input')
    if 'home' in keys:result.append('buff_input')
    return result or ['other_input' if keys else 'idle']


def input_intervals(events):
    """All labels describe input intent. None asserts a hit, kill or landing."""
    held=set();focus=False;last=None;censored=False;out=[]
    for t,batch in groupby(events,key=lambda e:e['t']):
        batch=list(batch)
        if last is not None and t>last and focus:
            gaps=[e.get('seconds',0) for e in batch if e['event']=='sampling_gap']
            out.append(dict(start=last,end=t,seconds=t-last,keys=sorted(held),tags=tags(held),
                            boundary_censored=censored or any(e['event'] in ('focus_lost','recording_end') for e in batch),
                            sampling_gap_seconds=max(gaps,default=0),source='human_observed_keys',result='unknown'))
        for e in batch:
            kind=e['event']
            if kind=='focus_gained':focus=True;held.clear();censored=True
            elif kind in ('focus_lost','recording_end'):focus=False;held.clear()
            elif kind in ('key_down','key_sync'):held.add(e['key'])
            elif kind in ('key_up','key_cancel'):held.discard(e['key'])
        if any(e['event'] in ('key_down','key_up') for e in batch):censored=False
        last=t
    return out


def analyze(folder,extract=True):
    folder=Path(folder);dest=folder/'analysis';dest.mkdir(exist_ok=True)
    events,event_bad=read_rows(folder/'inputs.jsonl');frames,frame_bad=read_rows(folder/'frames.jsonl')
    report=json.loads((folder/'report.json').read_text(encoding='utf-8')) if (folder/'report.json').exists() else {}
    intervals=input_intervals(events)
    with (dest/'input_intervals.jsonl').open('w',encoding='utf-8') as f:
        for row in intervals:f.write(json.dumps(row)+'\n')
    counts=Counter(e['key'] for e in events if e['event']=='key_down')
    durations=Counter()
    for row in intervals:
        for key in row['keys']:durations[key]+=row['seconds']
    grouped={}
    for row in frames:grouped.setdefault(row['segment'],[]).append(row)
    videos=[];issues=[]
    if not report:issues.append('缺少结束报告，可能异常退出。')
    if not events or events[-1]['event']!='recording_end':issues.append('缺少输入记录结束标记，最后一个按键片段可能不完整。')
    if event_bad or frame_bad:issues.append('日志存在不完整行；可能异常退出，不能作为完整示范。')
    if report.get('error'):issues.append('录制报告包含错误：'+report['error'])
    if report.get('encoder_drops',0) or report.get('capture_mailbox_gaps',0):issues.append('存在丢帧，分析动作时须使用原始采集时间。')
    gaps=[e['seconds'] for e in events if e['event']=='sampling_gap']
    if gaps:issues.append('按键轮询存在超过 30 ms 的间隔；受影响时段不作为精细时序真值。')
    if report.get('foreground_seconds',0)<480:issues.append('有效前台时间不足 8 分钟；可分析已有片段，但尚不足一轮十分钟基线。')
    for name,rows in grouped.items():
        path=(folder/name).resolve()
        if path.parent!=folder.resolve():raise ValueError('Video path escapes session folder')
        video=cv2.VideoCapture(str(path));n=int(video.get(cv2.CAP_PROP_FRAME_COUNT));ok,_=video.read()
        last_ok=False
        if n:video.set(cv2.CAP_PROP_POS_FRAMES,n-1);last_ok,_=video.read()
        video.release()
        good=ok and last_ok and n==len(rows)
        if not good:issues.append(f'{name} 的解码或帧索引未通过检查。')
        videos.append(dict(file=name,decoded_frame_count=n,indexed_frames=len(rows),first_and_last_readable=bool(ok and last_ok),valid=bool(good)))
    candidates=[];seen=set()
    for row in intervals:
        category=next((t for t in row['tags'] if t in ('rope_catch_attempt_input','drop_attempt_input','jump_input','attack_input')),None)
        if category and sum(x['category']==category for x in candidates)<3:
            if (category,round(row['start'])) in seen:continue
            candidates.append(dict(t=row['start'],category=category,result='requires_visual_review'))
            seen.add((category,round(row['start'])))
    for i,candidate in enumerate(candidates):
        if not frames:break
        row=min(frames,key=lambda r:abs(r['capture_started']-candidate['t']))
        candidate.update(segment=row['segment'],video_frame=row['video_frame'],frame_t=row['capture_started'])
        if extract:
            video=cv2.VideoCapture(str(folder/row['segment']));video.set(cv2.CAP_PROP_POS_FRAMES,row['video_frame']);ok,image=video.read();video.release()
            if ok:
                name=f'candidate_{i:02d}_{candidate["category"]}.jpg'
                cv2.imwrite(str(dest/name),image);candidate['image']=name
    def evidence_index(name):
        path=folder/name
        rows,bad=read_rows(path) if path.exists() else ([],0)
        return dict(samples=len(rows),invalid_lines=bad,
                    first_t=rows[0]['t'] if rows else None,last_t=rows[-1]['t'] if rows else None,
                    final_snapshot=bool(rows and rows[-1].get('kind')=='final'))
    result=dict(input_key_down_counts=dict(counts),observed_key_hold_seconds={k:round(v,3) for k,v in durations.items()},
                indexed_frames=len(frames),intervals=len(intervals),focus_loss_count=sum(e['event']=='focus_lost' for e in events),
                sampling_gaps=len(gaps),max_sampling_gap_seconds=max(gaps,default=0),invalid_log_lines=event_bad+frame_bad,
                video_checks=videos,quality_issues=issues,review_candidates=candidates,
                hud_evidence=evidence_index('hud.jsonl'),combat_evidence=evidence_index('combat.jsonl'),
                confirmed_hits=None,confirmed_kills=None,experience_gain=None,
                interpretation='Input-only labels are candidates, not proof of navigation/combat success.')
    atomic_json(dest/'analysis.json',result)
    (dest/'README.txt').write_text('录制质量与操作索引已生成。\n'
        'analysis.json：视频完整性、丢帧/采样间隙、按键统计及待复核片段。\n'
        'input_intervals.jsonl：人工操作的按键组合与持续时间；result=unknown，不推断命中、击杀或抓绳成功。\n'
        'candidate_*.jpg：关键按键附近的画面，结合根目录 review.html 查看前后动作。\n'+
        '\n'.join(issues or ['基础日志和视频检查通过；仍需视觉复核动作与成果。']),encoding='utf-8')
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='离线分析人工示范，不连接游戏')
    parser.add_argument('folder',type=Path);parser.add_argument('--no-images',action='store_true')
    args=parser.parse_args();result=analyze(args.folder,not args.no_images)
    print(json.dumps(result,ensure_ascii=False,indent=2))
