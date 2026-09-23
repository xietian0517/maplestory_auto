"""No real input: verify attack directions, paired movement and extreme settings."""
from unittest import TestCase, mock
from autofarm import plans
from autofarm.bot import BotStopped
from farm import Config


class DirectionProbabilityTests(TestCase):
    def test_threshold_assigns_actual_attacks(self):
        for probability, draws, expected in [
            (0, [0, .99], {'right': 0, 'left': 2}),
            (1, [0, .99], {'right': 2, 'left': 0}),
            (.7, [.69, .7], {'right': 1, 'left': 1}),
            (.7, [.1, .6], {'right': 2, 'left': 0}),
        ]:
            with self.subTest(probability=probability, draws=draws), \
                    mock.patch('autofarm.plans.random.random', side_effect=draws):
                self.assertEqual(plans.random_attack_counts(probability), expected)

    def test_invalid_probability_rejected(self):
        for value in (-.1, 1.1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                plans.random_attack_counts(value)

    def test_single_direction_still_moves_both_sides_and_extra_attacks_follow(self):
        for probability, selected in [(0, 'left'), (1, 'right')]:
            for order in [('left', 'right'), ('right', 'left')]:
                for extra in (0, 1):
                    with self.subTest(probability=probability, order=order, extra=extra):
                        bot = mock.Mock()
                        movement, attacks = [], []
                        current = [None]
                        def move(bot, side, seconds):
                            current[0] = side
                            movement.append((side, seconds))
                        def attack(*args):
                            attacks.append(current[0])
                        with mock.patch('autofarm.plans.maybe_buff', side_effect=[None, BotStopped()]), \
                             mock.patch('autofarm.plans.random_sides', return_value=order), \
                             mock.patch('autofarm.plans.random.random', return_value=.4), \
                             mock.patch('autofarm.plans.B.move', side_effect=move), \
                             mock.patch('autofarm.plans.B.jump_attack', side_effect=attack), \
                             mock.patch('autofarm.plans.B.settle'), \
                             mock.patch('autofarm.plans.B.wait'), \
                             mock.patch('autofarm.plans.B.switch_side'):
                            with self.assertRaises(BotStopped):
                                plans.random_jump(bot, Config(right_attack_prob=probability,
                                    extra_attack_prob=extra, hop_prob=0, idle_prob=0, buff_slots=()))
                        self.assertEqual(attacks, [selected] * (2 + 2 * extra))
                        self.assertEqual([s for s, _ in movement], list(order))
                        self.assertEqual(movement[0][1], movement[1][1])
