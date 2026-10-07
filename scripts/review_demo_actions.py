"""Build timestamped review sheets from saved video; never sends input."""
import argparse
import bisect
import json
from pathlib import Path

import cv2
from PIL import Image, ImageDraw


def main(folder):
    folder = Path(folder)
    out = folder/'analysis'/'actions'
    out.mkdir(parents=True, exist_ok=True)
    frames = [json.loads(s) for s in (folder/'frames.jsonl').read_text().splitlines()]
    times = [r['capture_started'] for r in frames]
    holds = json.loads((folder/'analysis'/'key_holds.json').read_text())
    caps = {}
    def frame(t):
        i = min(len(times)-1, bisect.bisect_left(times, t))
        if i and abs(times[i-1]-t)<abs(times[i]-t):
            i -= 1
        r = frames[i]
        cap = caps.setdefault(r['segment'], None)
        if cap is None:
            cap = caps[r['segment']] = cv2.VideoCapture(str(folder/r['segment']))
        cap.set(cv2.CAP_PROP_POS_FRAMES, r['video_frame'])
        ok, im = cap.read()
        if not ok:
            raise ValueError('Unreadable frame')
        return r, Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    specs = [(f'up_{i:02d}', [h['start']-.2, h['start']+.25, h['start']+.7, h['end']+.35])
             for i,h in enumerate(x for x in holds if x['key']=='up')]
    specs += [('combat_first', [7.4,7.72,7.95,8.24,8.5,8.9]),
              ('combat_ground', [23.4,23.7,24.0,24.3,24.6,24.9]),
              ('combat_upper', [58.6,58.85,59.05,59.25,59.5,59.8]),
              ('other_player', [567.5,568,568.5,569,569.5,570]),
              ('descent', [3.5,3.85,4.2,4.8,5.2,5.7])]
    index = []
    for name, stamps in specs:
        sheet = Image.new('RGB',(1366, ((len(stamps)+1)//2)*410),'#eeeeee')
        draw = ImageDraw.Draw(sheet)
        shots=[]
        for j,t in enumerate(stamps):
            r, im = frame(t)
            filename=f'{name}_{j}.png'
            im.save(out/filename)
            x,y = j%2*683, j//2*410
            sheet.paste(im.resize((683,384)), (x,y+24))
            draw.text((x+4,y+4),f"t={r['capture_started']:.3f} keys={'+'.join(r['keys'])}",fill='black')
            shots.append(dict(t=r['capture_started'],segment=r['segment'],video_frame=r['video_frame'],keys=r['keys'],image=filename))
        sheet.save(out/(name+'.jpg'),quality=93)
        index.append(dict(name=name,shots=shots,result='requires_visual_review'))
    for cap in caps.values():
        cap.release()
    (out/'index.json').write_text(json.dumps(index,indent=2),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder')
    main(parser.parse_args().folder)
