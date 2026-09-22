from pathlib import Path
import json
import tempfile
from unittest import TestCase, mock

import cv2
import numpy as np
from PIL import Image

from autofarm.custom_template import clipboard_image, prepare_image, save_template, check_owner
from autofarm.rope_archer import RopeScene
from name_template_gui import crop_box

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / 'templates/rope_archer/profile.json'


class ClipboardTemplateTests(TestCase):
    def setUp(self):
        self.pixels = np.random.default_rng(19).integers(0, 256, (14, 62, 3), np.uint8)
        self.image = Image.fromarray(self.pixels)

    def test_bitmap_clipboard_is_copied_without_resizing(self):
        with mock.patch('autofarm.custom_template.ImageGrab.grabclipboard', return_value=self.image):
            result = clipboard_image()
        np.testing.assert_array_equal(np.asarray(result), self.pixels)
        self.assertIsNot(result, self.image)

    def test_single_copied_image_file_can_be_loaded(self):
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / '角色名字.png'
            self.image.save(file)
            with mock.patch('autofarm.custom_template.ImageGrab.grabclipboard', return_value=[str(file)]):
                np.testing.assert_array_equal(np.asarray(clipboard_image()), self.pixels)

    def test_text_empty_or_multiple_files_do_not_replace_template(self):
        for value in (None, 'CatApril', [], ['a.png', 'b.png']):
            with mock.patch('autofarm.custom_template.ImageGrab.grabclipboard', return_value=value):
                with self.assertRaises(ValueError):
                    clipboard_image()

    def test_transparency_uses_same_visible_background(self):
        image = Image.new('RGBA', (20, 10), (255, 0, 0, 0))
        self.assertEqual(prepare_image(image).getpixel((0, 0)), (30, 30, 30))

    def test_save_preserves_pixels_and_reuses_identical_content(self):
        with tempfile.TemporaryDirectory() as folder:
            path = save_template(self.image, 'CatApril', folder)
            self.assertEqual(save_template(self.image, 'catapril', folder), path)
            self.assertNotEqual(save_template(self.image, 'OtherName', folder), path)
            with Image.open(Path(folder) / path) as saved:
                np.testing.assert_array_equal(np.asarray(saved), self.pixels)
            self.assertFalse(list(Path(folder).rglob('*.tmp')))

    def test_empty_or_uncropped_images_are_rejected_before_save(self):
        with tempfile.TemporaryDirectory() as folder:
            for image in (Image.new('RGB', (80, 14)), Image.new('RGB', (800, 600)),
                          Image.new('RGB', (4, 14))):
                with self.assertRaises(ValueError):
                    save_template(image, root=folder)
            self.assertFalse(list(Path(folder).iterdir()))

    def test_binding_change_is_rejected_including_unnamed_to_named(self):
        check_owner('catapril', 'CatApril')
        check_owner('', '')
        for name, owner in [('Other', 'CatApril'), ('CatApril', ''), ('', 'CatApril')]:
            with self.assertRaises(ValueError):
                check_owner(name, owner)

    def test_crop_coordinates_handle_reverse_drag_and_preview_scaling(self):
        self.assertEqual(crop_box((180, 60), (20, 20), (200, 100), (1000, 500)), (100, 100, 900, 300))
        self.assertEqual(crop_box((-10, -5), (220, 120), (200, 100), (1000, 500)), (0, 0, 1000, 500))


class CustomSceneTests(TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.frame = cv2.imread(str(ROOT / 'tests/fixtures/archer_monkey_sequence_6.png'))
        self.crop = self.frame[419:431, 462:511].copy()
        image = Image.fromarray(cv2.cvtColor(self.crop, cv2.COLOR_BGR2RGB))
        self.path = str(Path(self.folder.name) / save_template(image, 'CatApril', self.folder.name))

    def scene(self):
        return RopeScene(PROFILE, 'CatApril', self.path, 'CatApril')

    def test_custom_picture_works_without_loading_ocr(self):
        with mock.patch('autofarm.name_ocr.LocalNameOCR') as reader:
            o = self.scene().observe(self.frame)
            reader.assert_not_called()
        self.assertFalse(o.reason)
        self.assertEqual(o.player_source, '粘贴名字图片')
        self.assertEqual((o.player.x, o.player.y), (486.5, 445))

    def test_covered_custom_name_does_not_use_original_badge(self):
        scene = self.scene()
        self.assertIsNotNone(scene.observe(self.frame).player)
        self.frame[415:434, 450:520] = 30
        o = scene.observe(self.frame)
        self.assertIsNone(o.player)
        self.assertIn('粘贴', o.reason)

    def test_duplicate_picture_stops(self):
        self.frame[419:431, 350:399] = self.crop
        o = self.scene().observe(self.frame)
        self.assertIsNone(o.player)
        self.assertIn('多个', o.reason)

    def test_custom_picture_still_obeys_height_and_edge_checks(self):
        for x, y in ((493, 419), (462, 460)):
            frame = self.frame.copy()
            frame[415:434, 450:520] = 30
            frame[y:y+12, x:x+49] = self.crop
            o = self.scene().observe(frame)
            self.assertTrue(o.reason)

    def test_bad_binding_or_missing_file_cannot_revert_to_old_template(self):
        with self.assertRaises(ValueError):
            RopeScene(PROFILE, 'Other', self.path, 'CatApril')
        Path(self.path).unlink()
        with self.assertRaises((FileNotFoundError, ValueError)):
            self.scene()


class TemplateConfigTests(TestCase):
    def make_app(self):
        # 只使用配置方法，测试不会创建桌面窗口或访问真实剪贴板。
        from rope_archer_gui import ArcherApp
        app = mock.Mock()
        app.vars = {}
        for key, value in [('archer_name_template', 'old.png'), ('archer_template_owner', 'Old')]:
            var = mock.Mock()
            var.get.side_effect = lambda key=key: app.values[key]
            var.set.side_effect = lambda value, key=key: app.values.__setitem__(key, value)
            app.vars[key] = var
        app.values = {'archer_name_template': 'old.png', 'archer_template_owner': 'Old'}
        app._snapshot.side_effect = lambda: app.values.copy()
        return ArcherApp, app

    def test_template_config_is_saved_for_next_launch_and_clear_persists(self):
        cls, app = self.make_app()
        with tempfile.TemporaryDirectory() as folder:
            config = Path(folder) / 'settings.json'
            with mock.patch('farm_gui.CONFIG_FILE', str(config)):
                cls._set_name_template(app, 'new.png', 'New')
                self.assertEqual(json.loads(config.read_text(encoding='utf-8'))['archer_name_template'], 'new.png')
                cls._set_name_template(app, '', '')
                self.assertEqual(json.loads(config.read_text(encoding='utf-8'))['archer_name_template'], '')

    def test_failed_config_write_restores_previous_template(self):
        cls, app = self.make_app()
        with tempfile.TemporaryDirectory() as folder:
            with mock.patch('farm_gui.CONFIG_FILE', str(Path(folder) / 'settings.json')), \
                 mock.patch('rope_archer_gui.os.replace', side_effect=PermissionError('locked')):
                with self.assertRaises(PermissionError):
                    cls._set_name_template(app, 'new.png', 'New')
        self.assertEqual(app.values, {'archer_name_template': 'old.png', 'archer_template_owner': 'Old'})
