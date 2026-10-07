"""Workflow state and controls with a hidden Tk window; never sends game input."""
import json
from pathlib import Path
import tempfile
import threading
import tkinter as tk
import unittest
from unittest.mock import patch

import cv2
from autofarm.realtime.semantic import make_request,atomic_json
from autofarm.realtime.session import SessionMonitor
from game_ai_gui import Window,prepare_session,duration_seconds

FIXTURES=Path(__file__).parent/'fixtures'/'realtime'


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.master=tk.Tk();cls.master.withdraw()

    @classmethod
    def tearDownClass(cls):
        cls.master.destroy()

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.base=Path(self.temp.name)
        self.patch=patch('game_ai_gui.app_directory',return_value=self.base);self.patch.start()
        self.env=patch.dict('os.environ',{'OPENAI_API_KEY':'','DEEPSEEK_API_KEY':''});self.env.start()
        self.root=tk.Toplevel(self.master);self.root.withdraw();self.w=Window(self.root)
        self.folder=self.base/'captures'/'example';self.folder.mkdir(parents=True)
        self.request=make_request(cv2.imread(str(FIXTURES/'seed.png')),self.folder)
        self.scene=json.loads((FIXTURES/'scene.json').read_text(encoding='utf-8'))
        self.scene['request_id']=self.request['request_id']
        atomic_json(self.folder/'minimap_calibration.json',dict(request_id=self.request['request_id'],
                    scale=.0625,intercept=[11.90625,35.625]))

    def tearDown(self):
        self.w.close();self.env.stop();self.patch.stop();self.temp.cleanup()

    def scene_ready(self):
        atomic_json(self.folder/'scene.json',self.scene);self.w.select_folder(self.folder)

    def test_first_load_opens_screenshot_root(self):
        with patch('game_ai_gui.filedialog.askdirectory',return_value='') as dialog:
            self.w.load()
            self.assertEqual(Path(dialog.call_args.kwargs['initialdir']),self.base/'captures')

    def test_remembers_selected_directory_on_disk(self):
        self.w.select_folder(self.folder)
        self.assertEqual(self.w.initial_directory(),self.folder)
        saved=json.loads(self.w.preferences.read_text(encoding='utf-8'))
        self.assertEqual(Path(saved['last_scene_directory']),self.folder)
        self.w.folder=None;self.w.last_directory=Path(saved['last_scene_directory'])
        self.assertEqual(self.w.initial_directory(),self.folder)

    def test_request_only_is_waiting_not_processing_or_ready(self):
        self.w.select_folder(self.folder)
        self.assertIn('等待 AI 识图',self.w.state_title.get())
        self.assertIn('不会在后台自动调用',self.w.state_detail.get())
        self.assertTrue(self.w.start_button.instate(['disabled']))
        self.assertFalse(self.w.progress_running)
        with patch('game_ai_gui.run') as run,patch('game_ai_gui.WinApi') as api:
            self.w.start(True);run.assert_not_called();api.assert_not_called()

    def test_external_scene_automatically_enables_start(self):
        self.w.select_folder(self.folder);self.w.notice='识图请求已复制'
        atomic_json(self.folder/'scene.json',self.scene);self.w.refresh()
        self.assertIn('准备就绪',self.w.state_title.get())
        self.assertFalse(self.w.start_button.instate(['disabled']))
        self.assertIn('点击“开始挂机”',self.w.next_step.get())

    def test_wrong_request_or_low_confidence_disables_start(self):
        self.scene_ready()
        for key,value in (('request_id','wrong'),('confidence',.4)):
            invalid=dict(self.scene);invalid[key]=value
            atomic_json(self.folder/'scene.json',invalid);self.w.refresh()
            self.assertTrue(self.w.start_button.instate(['disabled']))
            self.assertIn('未通过检查',self.w.state_title.get())

    def test_corrupt_seed_is_not_ready_and_repaired_seed_is_detected(self):
        self.scene_ready();path=self.folder/self.request['image'];raw=path.read_bytes()
        path.write_bytes(b'broken');self.w.refresh()
        self.assertTrue(self.w.start_button.instate(['disabled']))
        path.write_bytes(raw);self.w.refresh()
        self.assertFalse(self.w.start_button.instate(['disabled']))

    def test_online_missing_configuration_is_explained(self):
        self.w.select_folder(self.folder);self.w.source.set('online');self.w.refresh()
        self.assertTrue(self.w.ai_button.instate(['disabled']))
        self.assertIn('未配置',self.w.state_detail.get())
        self.assertIn('当前对话',self.w.next_step.get())

    def test_action_api_applies_in_memory_without_saving_key(self):
        with patch.dict('os.environ',{'OPENAI_API_KEY':''}):
            self.w.set_action_api('offline-test-secret','test-vision-model')
            import os
            self.assertEqual(os.environ['OPENAI_API_KEY'],'offline-test-secret')
            self.assertEqual(self.w.action_model,'test-vision-model')
            self.assertNotIn('offline-test-secret',self.w.log.get('1.0','end'))
            self.assertNotIn('offline-test-secret',self.w.notice)
            for path in self.base.rglob('*.json'):
                self.assertNotIn('offline-test-secret',path.read_text(encoding='utf-8'))

    def test_action_api_rejects_missing_key_and_invalid_header(self):
        with patch.dict('os.environ',{'OPENAI_API_KEY':''}):
            for key,model in (('','test-model'),('two\nlines','test-model'),('abc','')):
                with self.subTest(key=key),self.assertRaises(ValueError):self.w.set_action_api(key,model)
            self.assertIsNone(self.w.action_model)

    def test_deepseek_configuration_diagnostic_and_online_scene_use_its_own_key(self):
        import os
        self.scene_ready()
        self.w.set_action_api('deepseek-offline-secret','deepseek-flash','deepseek')
        self.assertEqual(os.environ['DEEPSEEK_API_KEY'],'deepseek-offline-secret')
        self.assertEqual(os.environ['OPENAI_API_KEY'],'')
        self.w.source.set('online');self.w.refresh()
        self.assertFalse(self.w.ai_button.instate(['disabled']))
        self.assertFalse(self.w.start_button.instate(['disabled']))
        with patch('autofarm.realtime.policy.DeepSeekActionPolicy') as provider:
            provider.return_value.diagnose.return_value=dict(image_input_tested=True)
            result=self.w.diagnose_action_api()
        self.assertTrue(result['ok']);self.assertEqual(result['provider'],'deepseek')
        with patch('autofarm.realtime.scene_api.create_scene_planner') as planner:
            self.w.scene_planner();planner.assert_called_once_with('deepseek','deepseek-flash')
        for path in self.base.rglob('*.json'):
            self.assertNotIn('deepseek-offline-secret',path.read_text(encoding='utf-8'))

    def test_diagnostic_failure_is_saved_as_safe_actionable_code(self):
        from autofarm.realtime.policy_api import PolicyAPIError
        self.w.set_action_api('offline-test-secret','test-vision-model')
        with patch('autofarm.realtime.policy.OpenAIActionPolicy') as provider:
            provider.return_value.diagnose.side_effect=PolicyAPIError('insufficient_quota',429)
            result=self.w.diagnose_action_api()
        self.assertFalse(result['ok']);self.assertEqual(result['code'],'insufficient_quota')
        path=self.base/'action_api_diagnostic.json'
        self.assertNotIn('offline-test-secret',path.read_text(encoding='utf-8'))
        title,remedy=Window.describe_runtime(dict(action_policy=dict(last_model_error='PolicyAPIError',
            last_model_error_details=result,model_blocked=True)))
        self.assertIn('余额',title);self.assertIn('429',title);self.assertIn('暂停模型重试',remedy)

    def test_unknown_diagnostic_exception_details_do_not_leak(self):
        self.w.set_action_api('offline-test-secret','test-vision-model')
        with patch('autofarm.realtime.policy.OpenAIActionPolicy') as provider:
            provider.return_value.diagnose.side_effect=RuntimeError('offline-test-secret')
            result=self.w.diagnose_action_api()
        self.assertEqual(result['error_type'],'RuntimeError')
        self.assertNotIn('offline-test-secret',json.dumps(result))

    def test_active_mode_requires_api_before_creating_session_or_starting_input(self):
        self.scene_ready();self.w.policy_mode.set('AI 主动')
        self.w.start(True)
        self.assertIsNone(self.w.worker);self.assertIn('AI 配置',self.w.notice)
        self.assertFalse((self.base/'captures'/'runs').exists())

    def test_only_actual_planner_operation_shows_ai_processing(self):
        self.scene_ready();self.w.operation='plan';self.w.phase='planning';self.w.phase_note='等待在线模型返回'
        self.w.refresh()
        self.assertIn('AI 正在识图',self.w.state_title.get())
        self.assertTrue(self.w.progress_running)
        self.assertTrue(self.w.start_button.instate(['disabled']))
        self.w.operation=None

    def test_cancelled_countdown_never_runs_action(self):
        event=threading.Event();event.set();phases=[]
        self.assertFalse(Window.countdown(event,lambda *args:phases.append(args),'开始挂机'))
        self.assertEqual(len(phases),1)

    def test_old_status_does_not_replace_idle_ready_state(self):
        self.scene_ready()
        atomic_json(self.folder/'status.json',dict(mode='LIVE',reason='rope_ascend',phase='running'))
        self.w.refresh()
        self.assertIn('准备就绪',self.w.state_title.get());self.assertIsNone(self.w.runtime_status)

    def test_progress_describes_rope_braking(self):
        text,_=Window.describe_runtime(dict(reason='rope_brake'))
        self.assertIn('停稳',text)

    def test_duration_supports_hours_and_continuous_mode(self):
        for value,unit,expected in (('600','秒',600),('30','分钟',1800),('1.5','小时',5400),('0','小时',0)):
            self.assertEqual(duration_seconds(value,unit),expected)
        for value in ('-1','0.5','nan','inf','1小时',''):
            with self.assertRaises(ValueError):duration_seconds(value)

    def test_automatic_parking_explains_health_reason(self):
        for phase in ('parking','parked','parking_unconfirmed'):
            text,_=Window.describe_runtime(dict(phase=phase,health=dict(return_reason='health_unreadable_return')))
            self.assertIn('血量持续无法识别',text)

    def test_refresh_request_keeps_start_available_and_copies_update_path(self):
        self.scene_ready()
        make_request(cv2.imread(str(FIXTURES/'seed.png')),self.folder/'refresh')
        self.w.refresh(force=True)
        self.assertFalse(self.w.start_button.instate(['disabled']))
        self.w.copy_request()
        self.assertIn(str((self.folder/'refresh').resolve()),self.root.clipboard_get())

    def test_alignment_loss_is_not_reported_as_a_new_map(self):
        text,action=Window.describe_runtime(dict(reason='camera_or_map_changed'))
        self.assertIn('不代表已经换地图',text)
        self.assertIn('原场景重新定位',action)

    def test_run_copy_retains_calibration_world_and_parking_without_stop_markers(self):
        self.scene_ready()
        names=('world_geometry.json','minimap_calibration.json','parking.json','health.json','initial_motion.json',
               'identity_name.png','farm_plan.json','navigation_policy.json','platform_intent.json')
        for name in names:(self.folder/name).write_bytes(b'asset')
        for name in ('STOP','PARK','status.json'):(self.folder/name).write_bytes(b'old')
        target=prepare_session(self.folder,self.base/'new_run')
        for name in names:self.assertEqual((target/name).read_bytes(),b'asset')
        for name in ('STOP','PARK','status.json'):self.assertFalse((target/name).exists())
        climb=prepare_session(self.folder,self.base/'climb',climb=True)
        self.assertFalse((climb/'farm_plan.json').exists())
        self.assertFalse((climb/'navigation_policy.json').exists())

    def test_finish_requests_parking_without_emergency_stop(self):
        self.scene_ready();(self.folder/'parking.json').write_text('{}')
        self.w.operation='run';self.w.run_mode='挂机';self.w.finish()
        self.assertTrue((self.folder/'PARK').exists())
        self.assertFalse((self.folder/'STOP').exists())
        self.assertFalse(self.w.cancel.is_set());self.w.operation=None

    def test_preset_load_does_not_start_input_and_default_is_ten_minutes(self):
        self.scene_ready();target=self.base/'captures'/'monkey_forest_v046'
        prepare_session(self.folder,target)
        self.w.load_preset()
        self.assertEqual(self.w.folder,target.resolve())
        self.assertIsNone(self.w.worker);self.assertEqual(self.w.seconds.get(),'600')

    def test_completed_parking_status_is_visible_in_idle_gui(self):
        self.scene_ready();self.w.runtime_status=dict(phase='parking_unconfirmed')
        self.w.refresh();self.assertIn('未完成安全停靠',self.w.state_title.get())

    def test_missing_minimap_calibration_disables_start_and_repair_is_detected(self):
        self.scene_ready();path=self.folder/'minimap_calibration.json';raw=path.read_bytes();path.unlink()
        self.w.refresh();self.assertTrue(self.w.start_button.instate(['disabled']))
        self.assertIn('小地图',self.w.monitor.state.detail)
        path.write_bytes(raw);self.w.refresh()
        self.assertFalse(self.w.start_button.instate(['disabled']))


if __name__=='__main__':unittest.main()
