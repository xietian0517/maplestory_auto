"""Render an offline evidence report with indexed video chapters."""
import argparse
import html
import json
from pathlib import Path


def render(folder):
    folder=Path(folder);out=folder/'analysis'
    feedback=json.loads((out/'feedback.json').read_text(encoding='utf-8'))
    review=json.loads((out/'visual_review.json').read_text(encoding='utf-8'))
    # This page includes manually reviewed session-specific statements. Refuse
    # to silently reuse them for a different recording.
    if folder.name != 'demo_20260925_130158_91500' or review['session_id'] != folder.name or feedback['net']['net_exp'] != 23283:
        raise ValueError('This reviewed report belongs to demo_20260925_130158_91500 only')
    readings=[json.loads(s) for s in (out/'hud_readings.jsonl').read_text(encoding='utf-8').splitlines()]
    rows=''.join(f"<tr><td>{r['start']:.1f}–{r['end']:.1f} 秒</td><td>{r['net_exp']:,}</td><td>{r['exp_per_minute']:.0f}</td></tr>" for r in feedback['minute_windows'])
    net=feedback['net']
    points=' '.join(f"{40+r['t']*900/600:.1f},{190-(r['exp']-feedback['first']['exp'])*160/net['net_exp']:.1f}" for r in readings if r['confirmed'])
    chapters=''.join(f'<button data-time="{r["start"]}">{r["start"]:.1f}s · {html.escape(r["title"])}</button>' for r in review['episodes'])
    annotations=''.join(f'<article><h3>{html.escape(r["title"])}</h3><p>{r["start"]:.2f}–{r["end"]:.2f} 秒：{html.escape(r["finding"])}</p><p>改进：{html.escape(r["improvement"])}</p></article>' for r in review['episodes'])
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>十分钟人工示范复盘</title>
<style>body{font:17px/1.6 system-ui;background:#f4f7fa;color:#182d42;max-width:1250px;margin:24px auto;padding:0 22px}h1{font-size:30px}section,article{background:white;padding:20px;margin:18px 0;border-radius:12px}strong{color:#086f69}button,select{font:inherit;margin:4px;padding:7px;border:1px solid #b6c7cd;border-radius:6px;background:white;cursor:pointer}video{width:100%;background:#102030}table{border-collapse:collapse;width:100%}td,th{padding:7px;border-bottom:1px solid #dae4eb;text-align:left}#state{font-family:monospace}svg{width:100%}small{color:#4c6277}</style>
<h1>人工示范复盘 · 猴子森林 · 44 级</h1>
<p>本页只读取已保存的录像和日志，不连接游戏、不发送按键。</p>
<section><strong>净经验 +NET · 约 RATE 经验/分钟</strong><p>经验：10,134 → 33,417；HUD 覆盖 599.532 秒。完整运行计时 600.339 秒，按完整计时约 2,327 经验/分钟。所有寻找目标、爬绳和失误时间均保留。</p>
<svg viewBox="0 0 980 235" aria-label="累计净经验曲线"><path d="M40 20 V190 H940" fill="none" stroke="#a0b0bf"/><polyline points="POINTS" fill="none" stroke="#087f73" stroke-width="3"/><text x="40" y="220">0 秒</text><text x="865" y="220">600 秒</text><text x="45" y="23">+23,283</text></svg>
<p>20 段视频共 35,760 帧全部解码通过；2,906 张 HUD、990 张攻击原图全部可读。平均 59.62 FPS；12 帧编码丢失集中在开头一次约 216 ms 的缺口，早于首个游戏操作。</p>
<p><b>尚不能认定 AI 达到人工水平。</b>这是人工收益基线；没有自动策略独立运行的十分钟收益。短期伤害已完成局部视觉核验，整场去重伤害、精确有效伤害和击杀数量仍未知。</p></section>
<section><h2>按证据回看</h2><p>点击章节定位后按播放；时间与按键来自原始帧索引。切换片段不会发送游戏按键。</p>CHAPTERS<br><select id="parts"></select><button id="slow">¼ 速</button><button id="normal">正常速度</button><p id="state"></p><video id="video" controls preload="metadata"></video></section>
ANNOTATIONS
<section><h2>每分钟经验</h2><table><tr><th>原始时间</th><th>净增长</th><th>按实际区间折算每分钟</th></tr>ROWS</table><p>257 次经验数值上升不等于 257 次独立击杀；经验奖励包含会员附加项，攻击归因另行核对。</p></section>
<section><h2>测量边界</h2><p>HUD 字体用三张人工核对的原图校准，2,906 张均精确匹配并由相邻重复读数确认。这个读取器限定本次布局和字体，等级或字形改变时保留未知。不能把伤害飘字之和直接当作扣除过量伤害后的有效伤害。</p><p>约 570 秒后有其他玩家的雷电攻击，相关伤害必须隔离归因。全场经验增量是真实角色收益，但不能直接归因给某一击。</p><a href="human_review.md">详细复盘与改进优先级</a> · <a href="feedback.json">结构化收益</a> · <a href="visual_review.json">视觉标注</a> · <a href="audit/quality.json">完整质量检查</a></section>
<script src="../review_data.js"></script><script>
const v=document.querySelector('#video'),parts=document.querySelector('#parts'),state=document.querySelector('#state');
for(const s of DEMO.segments){const o=document.createElement('option');o.value=s.file;o.textContent=s.file+' · '+s.start.toFixed(1)+'–'+s.end.toFixed(1)+' 秒';parts.appendChild(o)}
function seek(file,frame=0){v.pause();parts.value=file;const s=DEMO.segments.find(x=>x.file===file);v.onloadedmetadata=()=>{v.currentTime=frame/s.fps};v.src='../'+file;}
parts.onchange=()=>seek(parts.value);if(DEMO.segments.length)seek(parts.value);
document.querySelectorAll('[data-time]').forEach(b=>b.onclick=()=>{const t=Number(b.dataset.time);let best=null;for(const s of DEMO.segments){for(const r of DEMO.frames[s.file]){if(!best||Math.abs(r.capture_started-t)<Math.abs(best.capture_started-t))best=r}}if(best)seek(best.segment,best.video_frame);});
document.querySelector('#slow').onclick=()=>v.playbackRate=.25;document.querySelector('#normal').onclick=()=>v.playbackRate=1;
function tick(){const s=DEMO.segments.find(x=>x.file===parts.value);if(s){const rows=DEMO.frames[s.file],r=rows[Math.min(rows.length-1,Math.floor(v.currentTime*s.fps))];if(r)state.textContent='原始时间 '+r.capture_started.toFixed(3)+' 秒 | 按住 '+(r.keys.join(' + ')||'无')+' | 原始帧 '+r.frame_id;}requestAnimationFrame(tick);}tick();
</script></html>'''
    for key,value in [('NET',f"{net['net_exp']:,}"),('RATE',f"{net['exp_per_minute']:,.0f}"),('POINTS',points),('CHAPTERS',chapters),('ANNOTATIONS',annotations),('ROWS',rows)]:
        page=page.replace(key,value)
    (out/'human_review.html').write_text(page,encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder');render(parser.parse_args().folder)
