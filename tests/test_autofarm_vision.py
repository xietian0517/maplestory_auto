"""名字牌定位与左右判断的测试：合成图 + 纯逻辑，不碰游戏窗口、不发按键。"""
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from autofarm import vision as V
from autofarm import plans
from farm import Config


class StopRun(Exception):
    """假 bot 用来结束无限循环。"""


class FakeBot:
    """只记录"往哪边挪"的机器人，不发按键。"""

    def __init__(self, move_limit):
        self.api = FakeApi()
        self.hwnd = 1
        self.moves = []
        self.limit = move_limit

    def log(self, message):
        pass

    def foreground(self):
        return True

    def gate(self):
        pass

    def wait(self, seconds):
        pass

    def down(self, key):
        pass

    def up(self, key):
        pass

    def tap(self, key, hold):
        pass

    def hold_key(self, key, seconds):
        self.moves.append(key)
        if len(self.moves) >= self.limit:
            raise StopRun


class FakeApi:
    def client_rect(self, hwnd):
        return dict(left=0, top=0, width=1000, height=600)


def make_frame():
    """造一张假画面：白字名字牌画在 (700, 300) 附近。"""
    frame = np.full((600, 1000, 3), 24, np.uint8)
    cv2.putText(frame, 'CatApril', (700, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def make_frame_with_name(text_x):
    """假画面：名字牌画在指定横坐标。"""
    frame = np.full((600, 1000, 3), 24, np.uint8)
    cv2.putText(frame, 'CatApril', (text_x, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (255, 255, 255), 2, cv2.LINE_AA)
    return frame


class DirectionTests(unittest.TestCase):
    """偏左就先打右、偏右就先打左；贴边一律往回，认不到就不改方向。"""

    def test_left_half_attacks_right(self):
        self.assertEqual(V.choose_direction(300, 200, 1200, 60), ('right', False))

    def test_right_half_attacks_left(self):
        self.assertEqual(V.choose_direction(900, 200, 1200, 60), ('left', False))

    def test_exact_center_goes_left(self):
        self.assertEqual(V.choose_direction(700, 200, 1200, 60), ('left', False))

    def test_near_left_end_rescues_inward(self):
        self.assertEqual(V.choose_direction(210, 200, 1200, 60), ('right', True))

    def test_near_right_end_rescues_inward(self):
        self.assertEqual(V.choose_direction(1190, 200, 1200, 60), ('left', True))

    def test_missing_position_keeps_caller_direction(self):
        self.assertEqual(V.choose_direction(None, 200, 1200, 60), (None, False))

    def test_bad_bounds_rejected(self):
        for left, right in ((None, 1200), (200, None), (1200, 200), (200, 200)):
            with self.assertRaises(ValueError):
                V.choose_direction(300, left, right, 60)


class NameFinderTests(unittest.TestCase):
    """模板匹配：坐标要对得上；空白画面和纯色模板都不能当成命中。"""

    @classmethod
    def setUpClass(cls):
        cls.frame = make_frame()
        cls.folder = Path(tempfile.mkdtemp())
        cls.template = cls.folder / 'name.png'
        cv2.imencode('.png', cls.frame[270:310, 690:830])[1].tofile(str(cls.template))

    def test_finds_name_and_returns_center(self):
        hit = V.NameFinder(str(self.template), 0.75).find(self.frame)
        self.assertIsNotNone(hit)
        self.assertAlmostEqual(hit.x, 760, delta=6)
        self.assertGreater(hit.score, 0.95)

    def test_blank_frame_is_not_a_hit(self):
        blank = np.full_like(self.frame, 24)
        self.assertIsNone(V.NameFinder(str(self.template), 0.75).find(blank))

    def test_low_texture_template_rejected(self):
        flat = self.folder / 'flat.png'
        cv2.imencode('.png', np.full((10, 10, 3), 30, np.uint8))[1].tofile(str(flat))
        with self.assertRaises(ValueError):
            V.NameFinder(str(flat), 0.75)

    def test_relative_template_resolves_under_program_dir(self):
        self.assertEqual(V.asset_path('assets/x.png'),
                         V.program_dir() / 'assets' / 'x.png')


class VisionJumpPlanTests(unittest.TestCase):
    """方案行为：判定在哪半边就一直往反方向打，跨过中线才换方向，不做左右各一下。"""

    @classmethod
    def setUpClass(cls):
        cls.folder = Path(tempfile.mkdtemp())
        cls.template = cls.folder / 'name.png'
        cv2.imencode('.png', make_frame()[270:310, 690:830])[1].tofile(str(cls.template))
        cls.right = make_frame_with_name(700)     # 名字 x≈760，画面中线 500 -> 右半边
        cls.left = make_frame_with_name(100)      # 名字 x≈160 -> 左半边

    def run_plan(self, frames, rounds):
        """跑 rounds 轮，返回每轮"往哪边挪"的列表。"""
        boxes = list(frames)
        state = {'i': 0}

        def fake_capture(region):
            frame = boxes[min(state['i'], len(boxes) - 1)]
            state['i'] += 1
            return frame

        bot = FakeBot(rounds)
        original = V.capture
        V.capture = fake_capture
        try:
            with self.assertRaises(StopRun):
                plans.vision_jump(bot, Config(name_template=str(self.template),
                                              vision_interval=0, edge_margin=60))
        finally:
            V.capture = original
        return bot.moves

    def test_one_move_per_round_no_back_and_forth(self):
        moves = self.run_plan([self.right], 4)
        self.assertEqual(moves, ['left'] * 4)

    def test_direction_keeps_until_side_changes(self):
        moves = self.run_plan([self.right] * 3 + [self.left] * 3, 6)
        self.assertEqual(moves, ['left', 'left', 'left', 'right', 'right', 'right'])

    def test_left_side_walks_right(self):
        moves = self.run_plan([self.left], 3)
        self.assertEqual(moves, ['right'] * 3)
