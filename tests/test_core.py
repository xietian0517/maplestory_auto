import copy
import json
from pathlib import Path
import unittest
from maplebot.core import Entity, Planner


class PlannerTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((Path(__file__).resolve().parents[1]/'config.example.json').read_text(encoding='utf-8'))
        self.p = Planner(self.cfg)

    def test_missing_player_never_moves(self):
        self.assertFalse(self.p.decide(None, [Entity(100,100)], None, 1).keys)

    def test_direction_then_attack_then_recovery(self):
        player, monsters = Entity(100,100), [Entity(300,100)]
        turn = self.p.decide(player, monsters, None, 1)
        self.assertEqual(turn.keys, ('right',))
        self.p.commit(turn, 1)
        attack = self.p.decide(player, monsters, None, 1.1)
        self.assertEqual(attack.keys, ('shift',))
        self.p.commit(attack, 1.1)
        self.assertEqual(self.p.decide(player, monsters, None, 1.2).reason, 'skill recovery')

    def test_priority_and_cooldown(self):
        attack = self.p.decide(Entity(100,100), [Entity(150,100)], None, 1)
        self.assertEqual(attack.keys, ('ctrl',))
        self.p.commit(attack, 1)
        self.p.ready['魔法双击'] = 10
        self.assertEqual(self.p.decide(Entity(100,100), [Entity(150,100)], None, 1.7).reason, 'cooldown')

    def test_cannot_chase_other_floor(self):
        self.assertFalse(self.p.decide(Entity(100,100), [Entity(150,250)], None, 1).keys)

    def test_stuck_stops_and_reset(self):
        self.p.decide(Entity(100,100), [Entity(500,100)], None, 1)
        self.assertFalse(self.p.decide(Entity(100,100), [Entity(500,100)], None, 7).keys)
        self.p.reset()
        self.assertEqual(self.p.decide(Entity(100,100), [Entity(500,100)], None, 8).keys, ('right',))

    def test_minimap_route_up_and_down(self):
        self.cfg['navigation']['waypoints'] = [{'x':20,'y':10,'vertical_keys':['up']}, {'x':20,'y':50,'vertical_keys':['down','space']}]
        p = Planner(self.cfg)
        self.assertEqual(p.decide(Entity(300,300), [], Entity(20,30), 1).keys, ('up',))
        p.decide(Entity(300,300), [], Entity(20,10), 2)
        self.assertEqual(p.decide(Entity(300,300), [], Entity(20,30), 3).keys, ('down','space'))

    def test_different_profession(self):
        cfg = copy.deepcopy(self.cfg)
        cfg['profile'] = 'warrior'
        p = Planner(cfg)
        p.facing = 'right'
        self.assertEqual(p.decide(Entity(0,0), [Entity(50,0)], None, 1).keys, ('x',))


if __name__ == '__main__':
    unittest.main()
