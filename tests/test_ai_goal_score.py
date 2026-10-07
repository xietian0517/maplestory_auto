"""Prevent short, rule-fallback or unverified runs from satisfying the AI goal."""
import json
from pathlib import Path
import tempfile
import unittest

from scripts.score_ai_goal_run import assess_ai


class GoalScoreTests(unittest.TestCase):
    def test_success_requires_all_live_duration_ai_and_reward_evidence(self):
        for missing in ('none','short','fallback','no_ai','no_attack','unverified','low_reward'):
            with self.subTest(missing=missing),tempfile.TemporaryDirectory() as folder:
                run=Path(folder)
                config=dict(mode='active',fallback='wait',provider='deepseek',model='deepseek-flash')
                counts=dict(responses=20,accepted=10,source_ai_executor=100,source_fallback_wait=300)
                report=dict(elapsed_seconds=600,attack_input_frames=80,action_policy=dict(counts=counts),keys_released=True)
                comparison=dict(complete_verified_run=True,target_met=True,automatic=dict(net_exp=20000))
                if missing=='short':report['elapsed_seconds']=134
                elif missing=='fallback':config['fallback']='rule'
                elif missing=='no_ai':counts['accepted']=0
                elif missing=='no_attack':report['attack_input_frames']=0
                elif missing=='unverified':comparison['complete_verified_run']=False
                elif missing=='low_reward':comparison['target_met']=False
                for name,data in [('action_policy_configuration.json',config),('run_configuration.json',dict(seconds=600)),
                        ('report.json',report)]:
                    (run/name).write_text(json.dumps(data),encoding='utf-8')
                result=assess_ai(run,comparison)
                self.assertEqual(result['target_met'],missing=='none')


if __name__=='__main__':unittest.main()
