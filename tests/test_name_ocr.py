"""文字定位和备用图的离线测试，不连接游戏、不发送按键。"""
from pathlib import Path
from unittest import TestCase, mock

import cv2
import numpy as np

from autofarm.name_ocr import LocalNameOCR, TextBox, TypedNameFinder, normalize_name, validate_name
from autofarm.rope_archer import RopeScene

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / 'templates/rope_archer/profile.json'


class TypedNameTests(TestCase):
    def setUp(self):
        self.reader = mock.Mock()
        self.reader.read.return_value = [TextBox('CatApril', .99, 20, 20, 80, 34)]
        self.finder = TypedNameFinder('CatApril', reader=self.reader)
        self.frame = np.full((70, 200, 3), 30, np.uint8)
        self.pattern = np.random.default_rng(23).integers(0, 256, (14, 60, 3), np.uint8)
        self.frame[20:34, 20:80] = self.pattern

    def test_exact_match_then_current_picture_tracks_without_more_ocr(self):
        hit = self.finder.find(self.frame, now=0)
        self.assertEqual((hit.x, hit.y), (50, 27))
        moved = np.full_like(self.frame, 30)
        moved[23:37, 47:107] = self.pattern
        hit = self.finder.find(moved, now=.1)
        self.assertEqual((hit.x, hit.y), (77, 30))
        self.assertEqual(self.finder.source, '名字图片（OCR确认）')
        self.reader.read.assert_called_once()

    def test_covered_name_never_reuses_previous_coordinates(self):
        self.finder.find(self.frame, now=0)
        self.assertIsNone(self.finder.find(np.full_like(self.frame, 30), now=.1))
        self.reader.read.return_value = []
        self.assertIsNone(self.finder.find(np.full_like(self.frame, 30), now=1))

    def test_similar_names_prefixes_pet_suffix_and_low_score_are_rejected(self):
        for text, score in [('CatApriI', .99), ('CatApril猴子', .99), ('CatApr', .99), ('CatApril', .94)]:
            with self.subTest(text=text, score=score):
                finder = TypedNameFinder('CatApril', reader=self.reader)
                self.reader.read.return_value = [TextBox(text, score, 20, 20, 80, 34)]
                self.assertIsNone(finder.find(self.frame, now=0))

    def test_chinese_spacing_and_english_case_are_normalized(self):
        self.assertEqual(normalize_name('冒 险 者'), '冒险者')
        self.reader.read.return_value = [TextBox('catapril', .99, 20, 20, 80, 34)]
        self.assertIsNotNone(self.finder.find(self.frame, now=0))

    def test_duplicate_ocr_names_or_pictures_stop(self):
        self.reader.read.return_value *= 2
        self.assertIsNone(self.finder.find(self.frame, now=0))
        self.assertTrue(self.finder.ambiguous)
        self.reader.read.return_value = self.reader.read.return_value[:1]
        self.finder.find(self.frame, now=1)
        self.frame[20:34, 115:175] = self.pattern
        self.assertIsNone(self.finder.find(self.frame, now=2))
        self.assertTrue(self.finder.ambiguous)

    def test_changed_name_has_no_previous_picture(self):
        self.finder.find(self.frame, now=0)
        other = TypedNameFinder('AnotherName', reader=self.reader)
        self.assertIsNone(other.find(self.frame, now=1))
        self.assertIsNone(other.template)

    def test_invalid_boxes_cannot_learn_a_template(self):
        for left, top, right, bottom in [(-2, 20, 80, 34), (20, 20, 500, 34),
                                         (80, 20, 20, 34), (20, 20, 80, float('nan'))]:
            self.reader.read.return_value = [TextBox('CatApril', .99, left, top, right, bottom)]
            self.assertIsNone(self.finder.find(self.frame, now=10 + len(str(right))))
        self.assertIsNone(self.finder.template)

    def test_name_validation_rejects_control_characters(self):
        for name in ('Cat\nApril', 'Cat\x00April', 'a' * 25):
            with self.assertRaises(ValueError):
                validate_name(name)


class SceneNameTests(TestCase):
    def scene(self, name, boxes=()):
        reader = mock.Mock()
        reader.read.return_value = list(boxes)
        with mock.patch('autofarm.name_ocr.LocalNameOCR', return_value=reader):
            return RopeScene(PROFILE, name), reader

    def test_unknown_typed_name_cannot_fall_back_to_catapril_badge(self):
        scene, _ = self.scene('AnotherName')
        frame = cv2.imread(str(ROOT / 'tests/fixtures/archer_monkey_sequence_6.png'))
        observation = scene.observe(frame)
        self.assertIsNone(observation.player)
        self.assertIn('AnotherName', observation.reason)

    def test_bound_name_can_use_picture_when_pet_covers_text(self):
        scene, reader = self.scene('CatApril')
        frame = cv2.imread(str(ROOT / 'tests/fixtures/archer_monkey_sequence_6.png'))
        for _ in range(3):
            observation = scene.observe(frame)
            self.assertFalse(observation.reason)
            self.assertIn('图片备用', observation.player_source)
        # 有备用图时，不让重复 OCR 暂停持续按键。
        reader.read.assert_called_once()

    def test_duplicate_text_does_not_fall_back_to_badge(self):
        boxes = [TextBox('CatApril', .99, x, 20, x+50, 32) for x in (50, 150)]
        scene, _ = self.scene('CatApril', boxes)
        observation = scene.observe(cv2.imread(str(ROOT / 'tests/fixtures/archer_monkey_sequence_6.png')))
        self.assertIsNone(observation.player)
        self.assertIn('多个', observation.reason)

    def test_blank_name_preserves_original_mode_without_loading_ocr(self):
        with mock.patch('autofarm.name_ocr.LocalNameOCR') as reader:
            scene = RopeScene(PROFILE, '')
            frame = cv2.imread(str(ROOT / 'tests/fixtures/archer_monkey_sequence_6.png'))
            self.assertEqual(scene.observe(frame).player_source, '完整称号牌')
            reader.assert_not_called()

    def test_text_position_obeys_horizontal_edge_only(self):
        for box in [TextBox('Other', .99, 265, 20, 290, 32),
                    TextBox('Other', .99, 180, 0, 220, 12)]:
            scene, _ = self.scene('Other', [box])
            o = scene.observe(cv2.imread(str(ROOT / 'tests/fixtures/archer_monkey_sequence_6.png')))
            self.assertEqual(bool(o.reason), box.left == 265)


class RealOCRTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reader = LocalNameOCR()

    def test_real_screenshot_reads_name_separately_from_pet(self):
        with mock.patch('autofarm.name_ocr.LocalNameOCR', return_value=self.reader):
            scene = RopeScene(PROFILE, 'CatApril')
        frame = cv2.imread(str(ROOT / 'tests/fixtures/archer_monkey_sequence_6.png'))
        observation = scene.observe(frame)
        self.assertFalse(observation.reason)
        self.assertEqual(observation.player_source, '名字文字 OCR')
        self.assertAlmostEqual(observation.player.x, 487, delta=3)
        self.assertAlmostEqual(observation.player.y, 444.5, delta=3)
        self.assertEqual(scene.observe(frame).player_source, '名字图片（OCR确认）')

    def test_actual_scene_does_not_accept_wrong_name(self):
        with mock.patch('autofarm.name_ocr.LocalNameOCR', return_value=self.reader):
            scene = RopeScene(PROFILE, 'CatApriI')
        o = scene.observe(cv2.imread(str(ROOT / 'tests/fixtures/archer_monkey_sequence_6.png')))
        self.assertIsNone(o.player)

    def test_model_can_read_chinese_badge_text(self):
        frame = cv2.imread(str(ROOT / 'tests/fixtures/archer_monkey_sequence_6.png'))
        boxes = self.reader.read(frame[396:452, 265:557])
        self.assertTrue(any('中级冒险家' in box.text and box.score > .95 for box in boxes))
