"""Offline integration tests: no game window and no real keyboard events."""
from dataclasses import replace
import json
import os
import queue
from pathlib import Path
import tempfile
from unittest import TestCase, mock

from autofarm.buffs import BuffScheduler, BuffSlot, parse_slots
from autofarm.rope_archer import ArcherController, RopeScene, Observation
from autofarm.vision import Hit
from farm import Config

ROOT = Path(__file__).resolve().parents[1]


class BuffTests(TestCase):
    def test_independent_timers_and_oldest_due_first(self):
        slots = (BuffSlot('home', (10, 10), (.3, .3)), BuffSlot('ins', (20, 20), (.3, .3)))
        with mock.patch('autofarm.buffs.time.monotonic', return_value=0) as clock:
            scheduler = BuffScheduler(Config(buff_slots=slots))
            bot = mock.Mock()
            bot.try_tap.return_value = True
            clock.return_value = 25
            self.assertTrue(scheduler.cast_one(bot, wait=False))
            self.assertEqual(bot.try_tap.call_args.args[0], 'home')
            self.assertEqual(scheduler.next_at, [35, 20])
            self.assertFalse(scheduler.cast_one(bot, wait=False))
            clock.return_value = 26
            self.assertTrue(scheduler.cast_one(bot, wait=False))
            self.assertEqual(bot.try_tap.call_args.args[0], 'ins')
            self.assertEqual(scheduler.next_at, [35, 46])

    def test_focus_or_pause_rejection_keeps_due_slot(self):
        with mock.patch('autofarm.buffs.time.monotonic', return_value=0) as clock:
            scheduler = BuffScheduler(Config(buff_slots=(BuffSlot('home', (1, 1), (0, 0)),)))
            clock.return_value = 100
            bot = mock.Mock()
            bot.try_tap.return_value = False
            self.assertFalse(scheduler.cast_one(bot))
            self.assertEqual(scheduler.next_at, [1])
            bot.wait.assert_not_called()
            bot.try_tap.return_value = True
            self.assertTrue(scheduler.cast_one(bot))
            self.assertFalse(scheduler.cast_one(bot))

    def test_invalid_intervals_hotkeys_and_conflicts_rejected(self):
        good = dict(key='ins', lo='60', hi='90', pmin='0', pmax='.6')
        self.assertEqual(parse_slots([good])[0].key, 'ins')
        for bad in [dict(key='f12'), dict(key='left'), dict(key='garbage'), dict(lo='nan'),
                    dict(hi='inf'), dict(lo='0'), dict(pmin='-1'), dict(lo='100')]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_slots([good | bad])
        with self.assertRaises(ValueError):
            parse_slots([good | {'key': 'shift'}], forbidden=('shift',))

    def test_disabled_and_legacy_single_slot(self):
        self.assertEqual(BuffScheduler(Config(buff_slots=())).slots, ())
        self.assertEqual(BuffScheduler(Config()).slots[0].key, 'home')

    def test_live_enable_disable_and_force_cast(self):
        with mock.patch('autofarm.buffs.time.monotonic', return_value=100):
            bot = mock.Mock()
            bot.buff_commands = queue.Queue()
            slot = BuffSlot('home', (60, 60), (.3, .3))
            scheduler = BuffScheduler(Config(buff_slots=()))
            bot.buff_commands.put(('replace', (slot,)))
            scheduler.sync(bot)
            self.assertTrue(scheduler.due())
            bot.try_tap.return_value = True
            self.assertTrue(scheduler.cast_one(bot, wait=False))
            self.assertEqual(scheduler.next_at, [160])
            bot.buff_commands.put(('replace', (slot,)))
            scheduler.sync(bot)
            self.assertEqual(scheduler.next_at, [160])
            bot.buff_commands.put(('now', None))
            scheduler.sync(bot)
            self.assertEqual(scheduler.next_at, [100])
            bot.buff_commands.put(('replace', ()))
            scheduler.sync(bot)
            self.assertFalse(scheduler.due())
            self.assertIn('关闭', bot.buff_status)

    def test_immediate_start_and_attack_release_preparation(self):
        with mock.patch('autofarm.buffs.time.monotonic', return_value=10):
            scheduler = BuffScheduler(Config(buff_start_immediately=True))
            self.assertTrue(scheduler.due())
            self.assertFalse(scheduler.prepare(10))
            self.assertFalse(scheduler.prepare(10.1))
            self.assertTrue(scheduler.prepare(10.2))
            scheduler.cancel_prepare()
            self.assertFalse(scheduler.prepare(11))


class ArcherBuffTests(TestCase):
    def setUp(self):
        self.c = ArcherController(RopeScene(ROOT / 'templates/rope_archer/profile.json').data)
        self.o = Observation(Hit(463, 522.5, 1), Hit(472, 202, 1), None)

    def test_buff_waits_for_two_valid_observations(self):
        action = self.c.decide(self.o, 0)
        self.assertFalse(self.c.can_buff(self.o, 0, action))
        action = self.c.decide(self.o, .1)
        self.assertTrue(self.c.can_buff(self.o, .1, action))

    def test_buff_rejected_when_missing_or_near_edge(self):
        for o in [replace(self.o, reason='missing'), replace(self.o, player=None),
                  replace(self.o, player=Hit(510, 522.5, 1)),
                  replace(self.o, player=Hit(400, 522.5, 1))]:
            self.c.reset()
            self.c.decide(o, 0)
            action = self.c.decide(o, .1)
            self.assertFalse(self.c.can_buff(o, .1, action))

    def test_buff_ignores_player_height(self):
        for y in (300, 505, 600, 700):
            self.c.reset()
            o = replace(self.o, player=Hit(463, y, 1))
            self.c.decide(o, 0)
            action = self.c.decide(o, .1)
            self.assertTrue(self.c.can_buff(o, .1, action))

    def test_buff_allowed_in_combat_only_after_right_turn_settles(self):
        o = replace(self.o, monkey=Hit(700, 516, 1))
        self.c.decide(o, 0)
        action = self.c.decide(o, .01)
        self.assertFalse(self.c.can_buff(o, .01, action))
        self.c.applied('right', o, .01)
        o = replace(o, player=Hit(469, 522.5, 1))
        action = self.c.decide(o, .06)
        self.assertFalse(self.c.can_buff(o, .06, action))
        action = self.c.decide(o, .16)
        self.assertEqual(action[0], 'shift')
        self.assertTrue(self.c.can_buff(o, .16, action))

    def test_run_releases_held_attack_before_buff_and_rescans(self):
        from autofarm.rope_archer import run
        from autofarm.bot import BotStopped
        bot, scene, controller, inputs, buffs = (mock.MagicMock() for _ in range(5))
        scene.data = RopeScene(ROOT / 'templates/rope_archer/profile.json').data
        scene.custom_player = scene.typed_name = None
        scene.observe.return_value = self.o
        controller.decide.return_value = ('shift', .4, 'attack')
        controller.can_buff.return_value = True
        inputs.__enter__.return_value = inputs
        inputs.epoch, inputs.key = 0, 'shift'
        bot.gate.side_effect = [False, BotStopped()]
        bot.paused = False
        bot.foreground.return_value = True
        buffs.cooling.return_value = False
        buffs.due.return_value = True
        order = []
        inputs.clear.side_effect = lambda: order.append('release')
        buffs.cast_one.side_effect = lambda *a, **k: (order.append('buff') or True)
        with mock.patch('autofarm.rope_archer.RopeScene', return_value=scene), \
             mock.patch('autofarm.rope_archer.ArcherController', return_value=controller), \
             mock.patch('autofarm.held_input.HeldInput', return_value=inputs), \
             mock.patch('autofarm.buffs.BuffScheduler', return_value=buffs), \
             mock.patch('autofarm.rope_archer.V.capture'), \
             mock.patch('autofarm.rope_archer.time.monotonic', return_value=10):
            with self.assertRaises(BotStopped):
                run(bot, Config(buff_slots=()))
        self.assertEqual(order, ['release', 'buff'])
        inputs.apply.assert_not_called()
        controller.reset.assert_not_called()
        controller.motion_stopped.assert_called_once_with(10)


class UnifiedGuiTests(TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        old = Path.cwd()
        os.chdir(self.folder.name)
        self.addCleanup(os.chdir, old)

    def app(self):
        from unified_gui import UnifiedApp
        import farm_gui
        old = farm_gui.CONFIG_FILE
        self.addCleanup(setattr, farm_gui, 'CONFIG_FILE', old)
        app = UnifiedApp()
        app.withdraw()
        self.addCleanup(app.destroy)
        return app

    def test_migration_preserves_old_files_and_new_settings_roundtrip(self):
        from unified_gui import read_settings, MODES
        files = {'rope_archer_config.json': {'archer_player_name': 'CatApril', 'archer_name_template': ''},
                 'gui_config.json': {'move_secs_min': '43'},
                 'buff_gui_config.json': {'slots': [dict(key='ins', lo='120', hi='180', pmin='.3', pmax='.6')]}}
        for name, data in files.items():
            Path(name).write_text(json.dumps(data), encoding='utf-8')
        app = self.app()
        self.assertEqual(app.vars['archer_player_name'].get(), 'CatApril')
        self.assertEqual(app.vars['move_secs_min'].get(), '43')
        for mode in MODES:
            app.mode_label.set(MODES[mode])
            app._select_mode()
            app.buff_on.set(True)
            cfg = app._build_config()
            self.assertEqual(cfg.plan, mode)
            self.assertEqual(cfg.buff_slots[0].key, 'ins')
        app._save()
        self.assertEqual(read_settings(Path.cwd())['plan'], 'buff_only')
        for name, original in files.items():
            self.assertEqual(json.loads(Path(name).read_text()), original)

    def test_disabled_buff_and_mode_switching_does_not_duplicate_worker(self):
        from unified_gui import MODES
        app = self.app()
        app.buff_on.set(False)
        app.mode_label.set(MODES['random_jump'])
        app._select_mode()
        self.assertEqual(app._build_config().buff_slots, ())
        app.mode_label.set(MODES['buff_only'])
        app._select_mode()
        self.assertEqual(len(app._build_config().buff_slots), 2)
        with mock.patch('farm_gui.threading.Thread') as thread:
            app.on_start()
            app.on_start()
            self.assertEqual(thread.call_count, 1)
            self.assertEqual(str(app.mode_box['state']), 'disabled')
            app._on_done()
            self.assertEqual(str(app.mode_box['state']), 'readonly')

    def test_direction_probability_default_complement_validation_and_save(self):
        from unified_gui import read_settings
        app = self.app()
        self.assertEqual(app._build_config().right_attack_prob, .5)
        self.assertEqual(app.left_probability.get(), '50')
        for right, left in [('70', '30'), ('0', '100'), ('100', '0'), ('62.5', '37.5')]:
            app.vars['right_attack_prob'].set(right)
            self.assertEqual(app.left_probability.get(), left)
            self.assertEqual(app._build_config().right_attack_prob, float(right) / 100)
        app._save()
        self.assertEqual(read_settings(Path.cwd())['right_attack_prob'], '62.5')
        restarted = self.app()
        self.assertEqual(restarted._build_config().right_attack_prob, .625)
        self.assertEqual(restarted.left_probability.get(), '37.5')
        for value in ('-1', '101', 'nan', 'inf', '', 'abc'):
            app.vars['right_attack_prob'].set(value)
            with self.assertRaises(ValueError):
                app._build_config()

    def test_buff_checkbox_updates_running_worker_and_persists(self):
        app = self.app()
        app.alive = True
        app.bot = mock.Mock()
        app.bot.buff_commands = queue.Queue()
        app.buff_on.set(True)
        app._apply_buffs()
        self.assertEqual(app.bot.buff_commands.get_nowait(), ('hold', .12))
        kind, slots = app.bot.buff_commands.get_nowait()
        self.assertEqual(kind, 'replace')
        self.assertEqual(len(slots), 2)
        self.assertTrue(json.loads(Path('unified_config.json').read_text(encoding='utf-8'))['buff_enabled'])
        app.buff_on.set(False)
        app._apply_buffs()
        app.bot.buff_commands.get_nowait()
        self.assertEqual(app.bot.buff_commands.get_nowait(), ('replace', ()))
