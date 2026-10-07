"""Copy verified demo glyphs into a portable, opt-in online EXP calibration."""
import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autofarm.realtime.demo_feedback import VerifiedHUDReader
from autofarm.realtime.semantic import load_scene, atomic_json


def prepare(demonstration, session):
    demo, folder = Path(demonstration), Path(session)
    scene, _ = load_scene(folder)
    config = json.loads((demo/'analysis/hud_calibration.json').read_text(encoding='utf-8'))
    VerifiedHUDReader(demo, config)
    with (demo/'hud.jsonl').open(encoding='utf-8') as log:
        first = json.loads(next(log))
    x, y, w, h = first['region']
    if x+w != scene.width or y+h != scene.height:
        raise ValueError('HUD calibration dimensions differ from the scene')
    dest = folder/'feedback_calibration'
    dest.mkdir(exist_ok=False)
    anchors = []
    for i, anchor in enumerate(config['anchors']):
        source = (demo/anchor['image']).resolve()
        if not source.is_relative_to(demo.resolve()):
            raise ValueError('Calibration anchor escapes demonstration')
        name = f'anchor_{i:03d}.png'
        shutil.copy2(source, dest/name)
        anchors.append({**anchor, 'image': name})
    portable = dict(config, anchors=anchors, level=None)
    portable.pop('level_reference', None)
    atomic_json(dest/'calibration.json', portable)
    VerifiedHUDReader(dest, portable)
    spec = dict(version=1, request_id=scene.request_id, hud_region=[x,y,x+w,y+h],
                calibration='feedback_calibration/calibration.json', interval=.2, max_gap=2,
                anchor_level_from_first_frame=True)
    atomic_json(folder/'feedback_config.json', spec)
    return spec


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('demonstration');p.add_argument('session');a=p.parse_args()
    print(json.dumps(prepare(a.demonstration, a.session), ensure_ascii=False, indent=2))
