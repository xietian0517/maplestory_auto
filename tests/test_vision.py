import json
from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np
from maplebot.vision import Vision, crop


class VisionTests(unittest.TestCase):
    def test_templates_offsets_monsters_and_minimap(self):
        cfg = json.loads((Path(__file__).resolve().parents[1]/'config.example.json').read_text(encoding='utf-8'))
        rng = np.random.default_rng(42)
        player = rng.integers(0,255,(16,12,3),dtype=np.uint8)
        monster = rng.integers(0,255,(18,14,3),dtype=np.uint8)
        frame = np.zeros((240,400,3), dtype=np.uint8)
        frame[100:116,70:82] = player
        frame[100:118,180:194] = monster
        frame[100:118,280:294] = monster
        frame[15:18,20:23] = (0,255,255)
        cfg.update(scene_roi=[50,60,340,170],minimap_roi=[0,0,50,40],player_templates=['p.png'],monster_templates=['m.png'])
        with tempfile.TemporaryDirectory() as folder:
            for name, data in [('p.png',player),('m.png',monster)]:
                cv2.imencode('.png', data)[1].tofile(str(Path(folder)/name))
            vision = Vision(cfg,folder)
            _, found, enemies, dot, outline = vision.read(frame)
            self.assertEqual((found.x,found.y), (26,48))
            self.assertEqual(len(enemies), 2)
            self.assertEqual((dot.x,dot.y), (21,16))
            self.assertEqual(outline.shape, (40,50,3))
            frame[150:166,70:82] = player
            self.assertIsNone(vision.read(frame)[1])

    def test_roi_bounds(self):
        with self.assertRaises(ValueError):
            crop(np.zeros((30,30,3),dtype=np.uint8), [20,20,20,20])


if __name__ == '__main__':
    unittest.main()
