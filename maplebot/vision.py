from pathlib import Path
import cv2
import numpy as np
from .core import Entity


def crop(frame, roi):
    if roi is None:
        return frame
    x, y, w, h = map(int, roi)
    if min(x, y) < 0 or min(w, h) <= 0 or x+w > frame.shape[1] or y+h > frame.shape[0]:
        raise ValueError('ROI 超出窗口范围，请重新标定')
    return frame[y:y+h, x:x+w]


class Vision:
    def __init__(self, config, base):
        self.config = config
        self.templates = {}
        for kind in ('player', 'monster'):
            self.templates[kind] = []
            for name in config[kind+'_templates']:
                template = cv2.imdecode(np.fromfile(Path(base)/name, dtype=np.uint8), cv2.IMREAD_COLOR)
                if template is None or template.std() < 5:
                    raise ValueError(f'模板无效或缺少纹理: {name}')
                self.templates[kind].extend([template, cv2.flip(template, 1)])

    def detect(self, scene, kind):
        candidates = []
        for template in self.templates[kind]:
            h, w = template.shape[:2]
            if h > scene.shape[0] or w > scene.shape[1]:
                continue
            scores = cv2.matchTemplate(scene, template, cv2.TM_CCOEFF_NORMED)
            for _ in range(50):
                _, score, _, (x, y) = cv2.minMaxLoc(scores)
                if not np.isfinite(score) or score < self.config['threshold']:
                    break
                candidates.append((score, x, y, w, h))
                scores[max(0,y-h//2):y+h//2+1, max(0,x-w//2):x+w//2+1] = -1
        selected = []
        for score, x, y, w, h in sorted(candidates, reverse=True):
            cx, cy = x+w/2, y+h/2
            if any(abs(cx-e.x) < w*0.6 and abs(cy-e.y) < h*0.6 for e in selected):
                continue
            selected.append(Entity(cx, cy, score))
        return selected

    def read(self, frame):
        scene = crop(frame, self.config['scene_roi'])
        players, monsters = self.detect(scene, 'player'), self.detect(scene, 'monster')
        # Multiple comparable matches are ambiguous: do not pick another player.
        player = players[0] if players and (len(players) == 1 or players[0].score-players[1].score > 0.05) else None
        dot, map_view = None, None
        if self.config.get('minimap_roi'):
            mini = crop(frame, self.config['minimap_roi'])
            lo, hi = self.config['minimap_player_hsv']
            mask = cv2.inRange(cv2.cvtColor(mini, cv2.COLOR_BGR2HSV), np.array(lo), np.array(hi))
            n, _, stats, centers = cv2.connectedComponentsWithStats(mask)
            dots = [centers[i] for i in range(1, n) if 2 <= stats[i, cv2.CC_STAT_AREA] <= 100]
            if len(dots) == 1:
                dot = Entity(*dots[0])
            # Only a visual contour estimate, not a collision/navigation mesh.
            edges = cv2.Canny(cv2.cvtColor(mini, cv2.COLOR_BGR2GRAY), 60, 140)
            map_view = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            if dot:
                cv2.circle(map_view, (int(dot.x), int(dot.y)), 4, (0,255,255), 1)
        return scene, player, monsters, dot, map_view
