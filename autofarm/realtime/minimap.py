"""Scrolling minimap observations and coarse route guidance.

Coordinates in the atlas are minimap pixels relative to the first accepted view,
not full-map coordinates. Low-resolution terrain is a hint, never a landing or
rope-grab authority. Only platforms confirmed by main-screen vision are routed.
"""
from collections import deque

import cv2
import numpy as np


def yellow_points(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (20, 130, 175), (38, 255, 255))
    _, _, stats, centers = cv2.connectedComponentsWithStats(mask)
    return [tuple(map(float, c)) for (_, _, w, h, a), c in zip(stats[1:], centers[1:])
            if 3 <= w <= 12 and 3 <= h <= 12 and 7 <= a <= 90 and a / (w * h) >= .3]


def locate_minimap(image):
    """Find the dark rectangular map pane, without fixed per-map coordinates."""
    roi = image[:min(420, image.shape[0]), :min(520, image.shape[1])]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(gray, 0, 110)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    choices = []
    for x, y, w, h, area in stats[1:]:
        if not (70 <= w <= 380 and 55 <= h <= 330 and x < 180 and y < 190):
            continue
        # Bright stone buildings occupy ~18% of the temple's real map pane.
        # Keep the UI-frame and unique-marker checks as the identity gates.
        if area / (w * h) < .75:
            continue
        pane = roi[y:y+h, x:x+w]
        if len(yellow_points(pane)) != 1:
            continue
        # The pane must be framed by light UI; dark scenery alone is insufficient.
        sides = [gray[max(0, y-6):y, x:x+w], gray[y+h:y+h+7, x:x+w],
                 gray[y:y+h, max(0, x-6):x], gray[y:y+h, x+w:x+w+7]]
        if sum(bool(s.size and np.mean(s > 155) > .25) for s in sides) < 3:
            continue
        choices.append((int(x), int(y), int(w), int(h)))
    return min(choices, key=lambda b: b[0]+b[1]) if choices else None


def marker_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Moving player, party and portal symbols must not anchor the terrain.
    mask = cv2.inRange(hsv, (0, 100, 170), (179, 255, 255))
    return cv2.dilate(mask, np.ones((7, 7), np.uint8))


def terrain_translation(previous, current):
    """Translation consensus across static patches, in either/both axes.

    Matching only overlapping patches permits a larger map to slide inside a
    stationary UI pane. Ambiguous, zoomed, hidden and unrelated views fail closed.
    """
    if previous.shape != current.shape:
        return None
    old = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    new = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    # The pane is translucent: the large main-camera background can move behind
    # stationary minimap terrain. Remove low frequencies before patch matching.
    old = old.astype(np.float32)-cv2.GaussianBlur(old, (0, 0), 3).astype(np.float32)
    new = new.astype(np.float32)-cv2.GaussianBlur(new, (0, 0), 3).astype(np.float32)
    excluded = marker_mask(previous)
    current_excluded = marker_mask(current)
    h, w = old.shape
    shifts, origins = [], []
    size = 16
    radius = min(28, min(h, w)//3)
    for y in np.linspace(3, h-size-3, 6).astype(int):
        for x in np.linspace(3, w-size-3, 6).astype(int):
            patch = old[y:y+size, x:x+size]
            if patch.std() < 9 or np.mean(excluded[y:y+size, x:x+size] > 0) > .08:
                continue
            x1, y1 = max(0, x-radius), max(0, y-radius)
            x2, y2 = min(w, x+size+radius), min(h, y+size+radius)
            scores = cv2.matchTemplate(new[y1:y2, x1:x2], patch, cv2.TM_CCOEFF_NORMED)
            _, score, _, (px, py) = cv2.minMaxLoc(scores)
            if score < .80:
                continue
            scores[max(0, py-2):py+3, max(0, px-2):px+3] = -1
            if score - cv2.minMaxLoc(scores)[1] < .035:
                continue
            nx, ny = px+x1, py+y1
            if np.mean(current_excluded[ny:ny+size, nx:nx+size] > 0) > .08:
                continue
            shifts.append((nx-x, ny-y)); origins.append((x, y))
    if len(shifts) < 4:
        return None
    shifts = np.array(shifts, dtype=float)
    votes = np.sum(np.max(abs(shifts[:, None]-shifts[None, :]), axis=2) <= 1, axis=1)
    keep = np.max(abs(shifts-shifts[np.argmax(votes)]), axis=1) <= 1
    if keep.sum() < 4 or keep.mean() < .55:
        return None
    spread = np.ptp(np.array(origins)[keep], axis=0)
    if spread[0] < w*.25 or spread[1] < h*.25:
        return None
    return np.median(shifts[keep], axis=0)


def terrain_hints(image):
    """Texture/line candidates for coarse exploration, not collision geometry."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    wood = cv2.inRange(hsv, (8, 70, 35), (35, 255, 185))
    wood[marker_mask(image) > 0] = 0
    wood = cv2.morphologyEx(wood, cv2.MORPH_CLOSE, np.ones((1, 3), np.uint8))
    wood = cv2.morphologyEx(wood, cv2.MORPH_OPEN, np.ones((1, 5), np.uint8))
    _, _, stats, _ = cv2.connectedComponentsWithStats(wood)
    platforms = [(float(x), float(x+w), float(y)) for x, y, w, h, a in stats[1:]
                 if w >= 6 and h <= 8 and w >= h*2 and a >= w]
    vines = cv2.inRange(hsv, (35, 90, 40), (85, 255, 200))
    vines[marker_mask(image) > 0] = 0
    vines = cv2.morphologyEx(vines, cv2.MORPH_OPEN, np.ones((6, 1), np.uint8))
    _, _, stats, _ = cv2.connectedComponentsWithStats(vines)
    ropes = [(float(x+w/2), float(y), float(y+h)) for x, y, w, h, _ in stats[1:]
             if w <= 2 and h >= 7 and any(a-2 <= x <= b+2 and abs(py-y) <= 4
                                        for a, b, py in platforms)]
    return platforms, ropes


class MinimapNavigator:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.calibration = None
        self.reference = None
        self.roi = None
        self.last_locate = -float('inf')
        self.generation = 0
        self.reason = 'searching' if enabled else 'disabled'
        self.reset()

    def configure(self, seed, data):
        """Bind reviewed world/minimap geometry to the seed's static terrain.

        Runtime player coordinates never come from a sprite or a name match.
        Reacquiring a pane must register against this reference before using the
        saved transform, including after a pause or a minimap scroll.
        """
        scale = float(data['scale'])
        intercept = np.asarray(data['intercept'], dtype=float)
        if not .015 <= scale <= .25 or intercept.shape != (2,) or not np.isfinite(intercept).all():
            raise ValueError('Invalid minimap calibration')
        roi = locate_minimap(seed)
        if roi is None:
            raise ValueError('Calibration seed has no unambiguous minimap')
        x, y, w, h = roi
        self.reference = seed[y:y+h, x:x+w].copy()
        self.calibration = (scale, intercept.copy())
        self.reset()
        self.roi = roi

    def reset(self):
        self.previous = None
        self.scroll = np.zeros(2)
        self.position = None
        self.marker = None
        self.stable = 0
        self.samples = deque(maxlen=120)
        self.scale = None
        self.intercept = None
        self.fit_at = None
        self.platforms = []
        self.ropes = []
        self.visits = {}
        self.trail = deque(maxlen=200)
        self.last_scan = -float('inf')
        self.last_visit = -float('inf')
        self.last_at = None
        self.generation += 1

    def observe(self, image, now):
        self.position = None
        self.marker = None
        if not self.enabled:
            return
        if self.roi is None:
            if now-self.last_locate < 1:
                return
            self.last_locate = now
            self.roi = locate_minimap(image)
            if self.roi is None:
                self.reason = 'not_found'
                return
            self.reset()
        x, y, w, h = self.roi
        pane = image[y:y+h, x:x+w]
        if pane.shape[:2] != (h, w):
            self.reset(); self.roi = None; self.reason = 'resized'
            return
        # Never stitch across focus pauses, recordings with gaps, or hidden panes.
        if self.last_at is not None and not 0 < now-self.last_at <= .6:
            self.reset()
        self.last_at = now
        if self.previous is None and self.calibration is not None:
            shift = terrain_translation(self.reference, pane)
            if shift is None:
                self.roi = None; self.reason = 'reference_mismatch'
                return
            self.scroll = shift.copy()
            self.scale, self.intercept = self.calibration
        if self.previous is not None:
            shift = terrain_translation(self.previous, pane)
            if shift is None:
                self.reset(); self.roi = None; self.reason = 'view_changed'
                return
            self.scroll += shift
            self.stable += 1
        self.previous = pane.copy()
        points = yellow_points(pane)
        if len(points) != 1:
            self.reason = 'marker_missing' if not points else 'marker_ambiguous'
            return
        self.marker = np.array(points[0])
        if self.stable < 2:
            self.reason = 'confirming'
            return
        self.position = self.marker-self.scroll
        if self.trail and np.linalg.norm(self.position-np.array(self.trail[-1])) > 15:
            self.reset(); self.reason = 'position_jump'
            return
        self.reason = 'tracking' if self.scale is not None else 'learning_scale'
        if self.calibration is not None:
            self.fit_at = now
        if now-self.last_visit >= .25:
            key = tuple(np.floor(self.position/5).astype(int))
            self.visits[key] = self.visits.get(key, 0)+1
            if len(self.visits) > 2048:
                del self.visits[next(iter(self.visits))]
            self.trail.append(self.position.tolist()); self.last_visit = now
        if now-self.last_scan >= .3:
            platforms, ropes = terrain_hints(pane)
            dx, dy = self.scroll
            self._merge(self.platforms, [(a-dx, b-dx, py-dy) for a, b, py in platforms])
            self._merge(self.ropes, [(a-dy, b-dy, px-dx) for px, a, b in ropes])
            self.last_scan = now

    @staticmethod
    def _merge(existing, segments):
        # Never bridge separate spans; overlapping observations extend coverage.
        for a, b, line in segments:
            match = next((i for i, (c, d, row) in enumerate(existing)
                          if abs(line-row) <= 1 and min(b, d)-max(a, c) >= 2), None)
            if match is None:
                existing.append((float(a), float(b), float(line)))
            else:
                c, d, row = existing[match]
                existing[match] = (min(a, c), max(b, d), row)
        del existing[:-256]

    def align(self, player, camera_offset, now):
        """Learn minimap pixels / main-world pixel from observed movement.

        Isotropic scaling is estimated, not hardcoded. Both coordinate residuals
        must agree; a changed zoom or disagreeing sprite clears the fit.
        """
        if self.calibration is not None or self.position is None or player.confidence < .8:
            return
        world = np.array([player.box.cx, player.box.y2])-camera_offset
        if self.scale is not None:
            error = np.linalg.norm(world*self.scale+self.intercept-self.position)
            if error > 3:
                self.samples.clear(); self.scale = None; self.intercept = None
                self.fit_at = None; self.reason = 'alignment_changed'
        if not self.samples or np.linalg.norm(world-self.samples[-1][0]) >= 12:
            self.samples.append((world.copy(), self.position.copy()))
        if len(self.samples) < 8:
            return
        world_points = np.array([p for p, _ in self.samples])
        mini_points = np.array([p for _, p in self.samples])
        if max(np.ptp(world_points, axis=0)) < 120 or max(np.ptp(mini_points, axis=0)) < 5:
            return
        wc, mc = world_points-world_points.mean(axis=0), mini_points-mini_points.mean(axis=0)
        scale = float(np.sum(wc*mc)/np.sum(wc*wc))
        intercept = np.median(mini_points-scale*world_points, axis=0)
        residual = np.linalg.norm(mini_points-(world_points*scale+intercept), axis=1)
        if not .015 <= scale <= .25 or np.percentile(residual, 90) > 1.5:
            self.scale = None; self.intercept = None; self.reason = 'alignment_uncertain'
            return
        self.scale = scale; self.intercept = intercept; self.fit_at = now
        self.reason = 'tracking'

    def ready(self, now):
        return (self.position is not None and self.scale is not None and self.fit_at is not None
                and now-self.fit_at < 1 and self.reason == 'tracking')

    def player_hint(self, camera_offset, now):
        if not self.ready(now):
            return None
        return tuple((self.position-self.intercept)/self.scale+camera_offset)

    def goals(self, platforms, player, camera_offset, now, bounds):
        """Rank main-view platforms by unexplored minimap terrain nearby.

        The ordinary controller still checks reachability and only executes its
        own screen-verified graph. Hints never add a foothold or a rope to it.
        """
        if not self.ready(now) or not self.platforms:
            return []
        current = [p for p in platforms if p.left <= player.box.cx <= p.right
                   and abs(p.y-player.box.y2) < 16]
        if not current or abs(player.vy) > 60:
            return []
        offset = np.array(camera_offset)
        def visited(point):
            key = np.floor(np.array(point)/5).astype(int)
            return sum(self.visits.get((key[0]+x, key[1]+y), 0)
                       for x in (-1, 0, 1) for y in (-1, 0, 1))
        # Wide, unseen surfaces are coarse exploration destinations. They may
        # extend off screen; only an observed first hop is returned below.
        targets = [np.array([(a+b)/2, y-3]) for a, b, y in self.platforms
                   if b-a >= 10 and visited(((a+b)/2, y-3)) < 3]
        if not targets:
            return []
        here = self.position
        floor = current[0]
        floor_left = (floor.left-offset[0])*self.scale+self.intercept[0]
        floor_right = (floor.right-offset[0])*self.scale+self.intercept[0]
        floor_y = (floor.y-offset[1])*self.scale+self.intercept[1]
        targets = [q for q in targets if not (floor_left-2 <= q[0] <= floor_right+2
                                              and abs(q[1]-floor_y) <= 4)]
        targets = sorted(targets, key=lambda q: np.linalg.norm(q-here))[:8]
        ranked = []
        for p in platforms:
            if p.id == current[0].id or p.right < bounds.x1+20 or p.left > bounds.x2-20:
                continue
            if not bounds.y1+10 <= p.y <= bounds.y2-10:
                continue
            point = (np.array([(p.left+p.right)/2, p.y])-offset)*self.scale+self.intercept
            costs = [np.linalg.norm(target-point)+.25*np.linalg.norm(target-here)
                     for target in targets if np.linalg.norm(target-point) < np.linalg.norm(target-here)-2]
            if not costs:
                continue
            ranked.append((min(costs)+min(30, visited(point))*2, p.id))
        return [ident for _, ident in sorted(ranked)]

    def status(self):
        return dict(state=self.reason, roi=list(self.roi) if self.roi else None,
                    generation=self.generation, scroll=self.scroll.tolist(),
                    position=self.position.tolist() if self.position is not None else None,
                    scale=self.scale, platform_hints=len(self.platforms), rope_hints=len(self.ropes))

    def to_data(self):
        return dict(**self.status(), coordinates='minimap pixels relative to first view; partial coverage',
                    platforms=[dict(left=a, right=b, y=y) for a, b, y in self.platforms],
                    ropes=[dict(x=x, top=a, bottom=b) for a, b, x in self.ropes],
                    trail=list(self.trail), terrain_role='exploration hints; not verified collision geometry')

    def annotate(self, image):
        if self.roi is None:
            return
        x, y, w, h = self.roi
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 220, 0), 1)
        if self.marker is not None:
            cv2.circle(image, (round(x+self.marker[0]), round(y+self.marker[1])), 5, (0, 255, 255), 1)
        for a, b in zip(self.trail, list(self.trail)[1:]):
            p = np.array(a)+self.scroll; q = np.array(b)+self.scroll
            if all(0 <= t[0] < w and 0 <= t[1] < h for t in (p, q)):
                cv2.line(image, (round(x+p[0]), round(y+p[1])), (round(x+q[0]), round(y+q[1])), (255, 255, 0), 1)
        cv2.putText(image, 'minimap: '+self.reason, (x, min(image.shape[0]-8, y+h+14)),
                    cv2.FONT_HERSHEY_SIMPLEX, .38, (0, 255, 255), 1)
