"""Conservative offline EXP feedback from manually calibrated, lossless HUDs.

This module has no capture, controller or Windows input dependency. An exact
glyph match is a measurement, not proof of which monster or attack earned EXP.
"""
import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np


def glyphs(image):
    mask = ((image.min(2) > 180) & (np.ptp(image.astype(int), axis=2) < 45)).astype(np.uint8)
    xs = np.where(mask.any(axis=0))[0]
    result = []
    for group in np.split(xs, np.where(np.diff(xs) > 1)[0] + 1):
        if not len(group):
            continue
        part = mask[:, group[0]:group[-1]+1]
        ys = np.where(part.any(axis=1))[0]
        part = part[ys[0]:ys[-1]+1]
        result.append((int(group[0]), part.shape, part.tobytes().hex()))
    return result


def crop(image, box):
    x1, y1, x2, y2 = box
    return image[y1:y2, x1:x2]


class VerifiedHUDReader:
    """Session-specific calibration; unrecognized pixels remain unknown.

    ``anchors`` supply the glyph alphabet (the font is identical across sessions).
    The level pixels, however, belong to the session that was calibrated: scoring a
    later run against the human session's level would reject every sample once the
    character has levelled up. ``level_reference`` points the same strict pixel
    comparison at the scored session's own level instead. A level change inside the
    scored run still fails that comparison, so the affected samples stay unknown.
    """
    def __init__(self, folder, calibration):
        self.folder = Path(folder)
        self.config = calibration
        self.alphabet = {}
        self.level_reference = None
        self.level_label = calibration.get('level')
        for anchor in calibration['anchors']:
            image = cv2.imread(str(self.folder / anchor['image']))
            if image is None:
                raise ValueError('Missing calibration image')
            self.shape = image.shape
            pieces = glyphs(crop(image, calibration['text_roi']))
            if len(pieces) != len(anchor['text']):
                raise ValueError('Calibration character count mismatch')
            for (_, shape, bits), char in zip(pieces, anchor['text']):
                key = (shape, bits)
                if key in self.alphabet and self.alphabet[key] != char:
                    raise ValueError('Ambiguous calibration glyph')
                self.alphabet[key] = char
            level = crop(image, calibration['level_roi'])
            if self.level_reference is None:
                self.level_reference = level
            elif not np.array_equal(level, self.level_reference):
                raise ValueError('Calibration level pixels disagree')
        reference = calibration.get('level_reference')
        if reference:
            path = Path(reference)
            image = cv2.imread(str(path if path.is_absolute() else self.folder / path))
            if image is None:
                raise ValueError('Missing level reference image')
            if self.level_reference is not None and image.shape != self.shape:
                raise ValueError('Level reference size mismatch')
            self.shape = image.shape
            self.level_reference = crop(image, calibration['level_roi'])

    def read(self, image):
        unknown = dict(exp=None, level=None, percent=None, reason='unrecognized_hud')
        if image is None or image.shape != self.shape:
            return unknown
        same_level = np.array_equal(crop(image, self.config['level_roi']), self.level_reference)
        # Without a numeric label this is still a verified constant level, not a guess.
        level = (self.level_label if self.level_label is not None else 'session') if same_level else None
        roi = crop(image, self.config['text_roi'])
        # Green square brackets separate the absolute number from percentage.
        b, g, r = (roi[:, :, i].astype(float) for i in range(3))
        green = (g > r * 1.15) & (g > b * 1.2) & (g > 100)
        xs = np.where(green.any(axis=0))[0]
        groups = [a for a in np.split(xs, np.where(np.diff(xs)>1)[0]+1) if len(a)]
        if len(groups) != 2:
            return {**unknown, 'level': level, 'reason': 'bracket_unknown'}
        digits, percent = '', ''
        for x, shape, bits in glyphs(roi):
            char = self.alphabet.get((shape, bits))
            if char is None:
                return {**unknown, 'level': level, 'reason': 'glyph_unknown'}
            if x < groups[0][0]:
                digits += char
            elif groups[0][-1] < x < groups[1][0]:
                percent += char
            else:
                return {**unknown, 'level': level, 'reason': 'layout_unknown'}
        if not digits.isdigit() or not re.fullmatch(r'\d{1,2}\.\d{2}%', percent):
            return {**unknown, 'level': level, 'reason': 'number_format_unknown'}
        return dict(exp=int(digits), level=level, percent=float(percent[:-1]),
                    reason='exact_calibrated_glyphs' if level is not None else 'level_changed_or_unknown')


def experience_delta(before, after, *, max_gap=None):
    """Never turn unknowns, negative deltas or a level rollover into rewards."""
    dt = after['t'] - before['t']
    if dt <= 0:
        return dict(net_exp=None, reason='nonpositive_time')
    if any(r.get('exp') is None or r.get('level') is None for r in (before, after)):
        return dict(net_exp=None, reason='unknown_reading')
    if before['level'] != after['level']:
        return dict(net_exp=None, reason='level_rollover_unknown')
    if max_gap is not None and dt > max_gap:
        return dict(net_exp=None, reason='observation_gap')
    delta = after['exp'] - before['exp']
    return dict(net_exp=delta, seconds=dt, exp_per_minute=delta * 60 / dt,
                reason='net_loss_review_required' if delta < 0 else 'same_level_net_change',
                attack_attribution='unknown')


def compare_runs(human, automatic):
    """An offline prediction/replay cannot meet a live return benchmark."""
    if automatic.get('mode') != 'LIVE_AUTOMATIC_OBSERVED':
        return dict(matched_human_exp_rate=False, reason='no_live_counterfactual_outcomes', score=None)
    if not automatic.get('independent_session') or not automatic.get('same_conditions'):
        return dict(matched_human_exp_rate=False, reason='unmatched_evaluation', score=None)
    if automatic.get('seconds', 0) < 590 or automatic.get('net_exp') is None or human.get('net_exp') is None:
        return dict(matched_human_exp_rate=False, reason='incomplete_outcomes', score=None)
    ratio = (automatic['net_exp']/automatic['seconds']) / (human['net_exp']/human['seconds']) if human['net_exp'] > 0 else None
    return dict(matched_human_exp_rate=bool(ratio is not None and ratio >= 1 and automatic.get('manual_interventions') == 0),
                reason='single_run_comparison_only', score=ratio)


def input_statistics(events):
    starts = {}
    holds = []
    downs = {}
    for e in events:
        kind = e['event']
        if kind == 'key_down':
            starts[e['key']] = e['t']
            downs[e['key']] = downs.get(e['key'], 0) + 1
        elif kind == 'key_up' and e['key'] in starts:
            holds.append(dict(key=e['key'], start=starts.pop(e['key']), end=e['t'], censored=False))
        elif kind in ('focus_lost', 'recording_end'):
            for key, start in starts.items():
                holds.append(dict(key=key, start=start, end=e['t'], censored=True))
            starts.clear()
    durations = {}
    for key in downs:
        values = [h['end']-h['start'] for h in holds if h['key']==key and not h['censored']]
        durations[key] = dict(key_downs=downs[key], complete_holds=len(values),
                              seconds_percentiles={str(q):float(np.percentile(values,q)) for q in (10,50,90)} if values else {})
    attacks = [h for h in holds if h['key']=='shift']
    delays = []
    for h in holds:
        if h['key'] != 'alt' or h['censored']:
            continue
        following = next((a for a in attacks if h['start']<=a['start']<=h['start']+.5),None)
        if following:
            delays.append(following['start']-h['start'])
    return dict(holds=holds, keys=durations, jump_then_attack_candidates=len(delays),
                jump_to_attack_seconds_percentiles={str(q):float(np.percentile(delays,q)) for q in (10,50,90)} if delays else {},
                interpretation='Observed input only; not proof of jump, rope grab, damage or kill.')


def analyze_feedback(folder, calibration_path):
    folder = Path(folder)
    calibration = json.loads(Path(calibration_path).read_text(encoding='utf-8'))
    reader = VerifiedHUDReader(folder, calibration)
    index = [json.loads(s) for s in (folder/'hud.jsonl').read_text(encoding='utf-8').splitlines()]
    readings = [dict(t=r['t'], image=r['image'], **reader.read(cv2.imread(str(folder/r['image'])))) for r in index]
    # Require a repeated identical EXP/level within 0.6 s; isolated readings
    # stay unconfirmed even if their individual glyphs are recognizable.
    for i, r in enumerate(readings):
        neighbors = readings[max(0, i-1):i] + readings[i+1:i+2]
        r['confirmed'] = r['exp'] is not None and r['level'] is not None and any(
            n['exp'] == r['exp'] and n['level'] == r['level'] and abs(n['t']-r['t']) <= .6 for n in neighbors)
    valid = [r for r in readings if r['confirmed']]
    summary = dict(mode='OFFLINE_HUMAN_EVIDENCE', automatic_input=False, samples=len(readings),
                   recognized=sum(r['exp'] is not None for r in readings), confirmed_samples=len(valid),
                   damage_total=None, confirmed_kills=None, damage_reason='requires_visual_attribution_and_deduplication')
    events = [json.loads(s) for s in (folder/'inputs.jsonl').read_text(encoding='utf-8').splitlines()]
    inputs = input_statistics(events)
    summary['input_statistics'] = {k:v for k,v in inputs.items() if k != 'holds'}
    if valid:
        before, after = valid[0], valid[-1]
        summary.update(first=before, last=after, net=experience_delta(before, after))
        summary['changes'] = [dict(t=b['t'], previous_t=a['t'], image=b['image'], **experience_delta(a, b, max_gap=.65))
                              for a,b in zip(valid, valid[1:]) if a['exp'] != b['exp'] or b['t']-a['t']>.65]
        summary['minute_windows'] = []
        for start in range(0, 600, 60):
            a = min(valid, key=lambda r:abs(r['t']-start))
            b = min(valid, key=lambda r:abs(r['t']-(start+60)))
            summary['minute_windows'].append(dict(start=a['t'], end=b['t'], before_image=a['image'], after_image=b['image'], **experience_delta(a,b)))
    out = folder/'analysis'
    out.mkdir(exist_ok=True)
    (out/'key_holds.json').write_text(json.dumps(inputs['holds'], indent=2), encoding='utf-8')
    (out/'hud_readings.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in readings), encoding='utf-8')
    (out/'feedback.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder')
    parser.add_argument('calibration')
    args = parser.parse_args()
    result = analyze_feedback(args.folder, args.calibration)
    print(json.dumps({k:v for k,v in result.items() if k != 'changes'}, indent=2))
