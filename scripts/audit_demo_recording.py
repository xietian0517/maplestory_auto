"""Decode and inspect saved demonstration files only. Never opens the game."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


def main(folder):
    folder = Path(folder)
    out = folder / 'analysis' / 'audit'
    out.mkdir(parents=True, exist_ok=True)
    def rows(name):
        return [json.loads(line) for line in (folder / name).read_text(encoding='utf-8').splitlines()]
    frames = rows('frames.jsonl')
    grouped = defaultdict(list)
    for r in frames:
        grouped[r['segment']].append(r)
    checks = []
    thumbnails = []
    next_overview = 0
    # Decode every frame, including the middle of each segment.
    for name, index in grouped.items():
        cap = cv2.VideoCapture(str(folder / name))
        n = 0
        while True:
            ok, im = cap.read()
            if not ok:
                break
            if n < len(index):
                r = index[n]
                if r['capture_started'] >= next_overview:
                    thumb = Image.fromarray(cv2.cvtColor(cv2.resize(im, (683, 384)), cv2.COLOR_BGR2RGB))
                    thumbnails.append((r, thumb))
                    next_overview += 5
            n += 1
        cap.release()
        checks.append(dict(segment=name, decoded=n, indexed=len(index), valid=n == len(index),
                           contiguous_index=all(r['video_frame'] == i for i, r in enumerate(index))))
        print(name, n, flush=True)
    for start in range(0, len(thumbnails), 12):
        sheet = Image.new('RGB', (1366, 6 * 410), '#eeeeee')
        draw = ImageDraw.Draw(sheet)
        for j, (r, im) in enumerate(thumbnails[start:start + 12]):
            x, y = j % 2 * 683, j // 2 * 410
            sheet.paste(im, (x, y + 24))
            draw.text((x + 5, y + 4), f"t={r['capture_started']:.3f} keys={'+'.join(r['keys'])}", fill='black')
        sheet.save(out / f'overview_{start//12:02d}.jpg', quality=92)
    evidence = {}
    for kind in ('hud', 'combat'):
        index = rows(kind + '.jsonl')
        bad = []
        for r in index:
            im = cv2.imread(str(folder / r['image']))
            if im is None or im.shape[1] != 1366 or im.shape[0] != (110 if kind == 'hud' else 768):
                bad.append(r['image'])
        evidence[kind] = dict(indexed=len(index), unreadable_or_wrong_size=bad)
    times = np.array([r['capture_started'] for r in frames])
    ages = [r['key_sample_age'] for r in frames]
    result = dict(mode='OFFLINE_FULL_DECODE', automatic_input=False, videos=checks, evidence=evidence,
                  frames=len(frames), first_t=float(times[0]), last_t=float(times[-1]),
                  frame_interval_ms={str(q): float(np.percentile(np.diff(times)*1000, q)) for q in (50, 95, 99, 100)},
                  key_sample_age_ms={str(q): float(np.percentile(ages, q)*1000) for q in (50, 95, 99, 100)},
                  gaps_over_50ms=[dict(before=float(times[i]), after=float(times[i+1]), seconds=float(times[i+1]-times[i]))
                                  for i in range(len(times)-1) if times[i+1]-times[i] > .05])
    (out / 'quality.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k != 'videos'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder')
    main(parser.parse_args().folder)
