"""Hidden-window workflow tests; do not attach to the real game."""
from pathlib import Path
import tempfile
import tkinter as tk
import unittest
from unittest.mock import patch

from game_demo_gui import DemoWindow,check_window


class DemoGUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.master=tk.Tk();cls.master.withdraw()
    @classmethod
    def tearDownClass(cls):cls.master.destroy()
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.base=Path(self.temp.name)
        self.patch=patch('game_demo_gui.app_directory',return_value=self.base);self.patch.start()
        self.root=tk.Toplevel(self.master);self.root.withdraw();self.w=DemoWindow(self.root)
    def tearDown(self):self.w.close();self.patch.stop();self.temp.cleanup()

    def test_default_ten_minutes_sixty_fps_and_no_autostart(self):
        self.assertEqual(self.w.minutes.get(),'10');self.assertEqual(self.w.fps.get(),'60')
        self.assertIsNone(self.w.worker);self.assertIsNone(self.w.folder)

    def test_focus_loss_displays_pause_and_actual_metrics(self):
        self.w.show_status(dict(phase='recording',elapsed_seconds=30,requested_seconds=600,
                               foreground=False,foreground_seconds=25,recent_capture_fps=0,target_fps=60,
                               frames_written=1400,encoder_drops=2,capture_mailbox_gaps=3,backend='fixture'))
        self.assertIn('暂停采集',self.w.title.get());self.assertIn('0.0 FPS',self.w.stats.get())
        self.assertIn('编码丢帧 2',self.w.stats.get())

    def test_copy_message_contains_current_recording_directory(self):
        self.w.folder=self.base/'demo';self.w.folder.mkdir();self.w.copy()
        prompt=self.root.clipboard_get()
        self.assertIn(str(self.w.folder.resolve()),prompt);self.assertIn('不要自动启动游戏按键',prompt)

    def test_auto_controller_conflict_is_visible(self):
        self.w.show_status(dict(phase='error',error='旧挂机程序仍在运行'))
        self.assertEqual(self.w.title.get(),'录制未完成');self.assertIn('旧挂机程序',self.w.detail.get())

    def test_partial_offscreen_window_is_rejected(self):
        class Api:
            def __init__(self):self.u=self
            def is_iconic(self,hwnd):return False
            def client_rect(self,hwnd):return dict(left=-333,top=0,width=1366,height=768)
            def GetSystemMetrics(self,i):return {76:0,77:0,78:1920,79:1080}[i]
        with self.assertRaisesRegex(ValueError,'屏幕外'):check_window(Api(),1)


if __name__=='__main__':unittest.main()
