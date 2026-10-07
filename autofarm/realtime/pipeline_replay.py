"""Deterministic observed-state replay; never imports a Windows input adapter."""
import json
from pathlib import Path

from .actions import ActionIntent
from .feedback import observation_from_data
from .model import Box, Decision
from .pipeline import DecisionPipeline, restore_controller, CONFIG_FIELDS
from .policy import PolicyConfig
from .semantic import atomic_json


def update_configuration(pipeline, data):
    """Apply a recorded semantic refresh to the existing rule state machine."""
    controller, base = pipeline.rule.controller, pipeline.base
    for name, value in data.get('values', {}).items():
        if name not in CONFIG_FIELDS:
            raise ValueError('Unknown recorded configuration field')
        if name in ('safe_platforms', 'firing_platforms', 'combat_platforms') and value is not None:
            value = set(value)
        setattr(base, name, value)
    controller.preferred = data.get('preferred', [])
    base.hud_boxes = [Box(**b) for b in data.get('hud_boxes', [])]
    if data.get('platform_policy'):
        from .demo_policy import PlatformIntent
        base.platform_policy = PlatformIntent(data['platform_policy'])
    else:
        base.platform_policy = None


def replay_trace(folder, output=None):
    folder = Path(folder)
    path = folder/'policy_trace.jsonl'
    pipeline = None
    cycles = matching = 0
    mismatches = []
    acknowledged = 0
    elapsed_start = elapsed_end = None
    events = 0
    terminal_recorded = False
    terminal_matches = None
    try:
        with path.open(encoding='utf-8') as rows:
            for line in rows:
                row = json.loads(line)
                if row.get('kind') == 'configuration':
                    if pipeline is not None or row.get('version') != 1:
                        raise ValueError('Invalid trace configuration header')
                    config = PolicyConfig.parse(row['policy'])
                    data = row['controller']
                    pipeline = DecisionPipeline(restore_controller(data), config, data['navigate'],
                                                data['climb'], replay=True)
                    continue
                if pipeline is None:
                    raise ValueError('Trace has no configuration header')
                if terminal_recorded:
                    raise ValueError('Trace has records after session end')
                if row.get('kind') == 'session_end':
                    pipeline.close(row['t'])
                    terminal_recorded = True
                    terminal_matches = row['events'] == pipeline.cycle_events
                    events += len(row['events'])
                    if not terminal_matches and len(mismatches)<20:
                        mismatches.append(dict(kind='session_end',t=row['t'],
                            expected_events=row['events'],actual_events=pipeline.cycle_events))
                    continue
                if row.get('kind') == 'reset':
                    pipeline.reset(row['t'], row['reason'])
                    continue
                if row.get('kind') != 'cycle':
                    raise ValueError('Unknown trace record')
                if row.get('configuration'):
                    update_configuration(pipeline, row['configuration'])
                o = observation_from_data(row['observation'])
                override = Decision(frozenset(row['override']['keys']), row['override']['reason'],
                                    row['override']['target']) if row.get('override') else None
                intent = ActionIntent(**row['delivered_intent']) if row.get('delivered_intent') else None
                d = pipeline.step(o, row['t'], offset=tuple(row['offset']), scene_id=row['scene_id'],
                    health=row.get('health'), override=override, override_source=row.get('override_source'),
                    releases=row.get('releases', []), delivery=intent, readings=row.get('hud_delivery'),
                    width=row.get('width', 1366))
                proposed=dict(keys=sorted(d.keys),reason=d.reason,target=d.target)
                same_proposal=proposed==row.get('proposed_decision',row['decision'])
                if row['source']=='frame_guard':
                    final=row['decision']
                    if final['keys']:
                        raise ValueError('Recorded frame guard must release input')
                    d=Decision(frozenset(),final['reason'],final['target'])
                receipt = row['input']
                pipeline.commit(d, receipt['applied'], receipt['t'], o,
                    receipt['held_keys'], simulated=receipt['simulated'],
                    external_owner=row.get('input_owner')=='external')
                expected, actual = row['decision'], pipeline.last_trace['decision']
                same = same_proposal and expected == actual and row['source'] == pipeline.last_trace['source']
                # Compare observed result events, excluding live transport telemetry.
                kinds = {'action_result', 'landing_observed', 'target_lost', 'hp_decreased',
                         'movement_no_observed_progress', 'experience_unknown', 'experience_changed',
                         'experience_baseline_confirmed'}
                expected_events = [e for e in row.get('events', []) if e.get('event') in kinds]
                actual_events = [e for e in pipeline.last_trace['events'] if e.get('event') in kinds]
                same = same and expected_events == actual_events
                matching += same
                if not same and len(mismatches) < 20:
                    mismatches.append(dict(cycle=cycles, t=row['t'], expected=expected, actual=actual,
                        expected_proposal=row.get('proposed_decision',expected),actual_proposal=proposed,
                        expected_source=row['source'], actual_source=pipeline.last_trace['source'],
                        expected_events=expected_events, actual_events=actual_events))
                acknowledged += row['acknowledged']
                events += len(expected_events)
                cycles += 1
                if elapsed_start is None:
                    elapsed_start = row['t']
                elapsed_end = row['t']
    finally:
        if pipeline:
            pipeline.close(elapsed_end or 0)
    if pipeline is None or not cycles:
        raise ValueError('Trace has no observed cycles')
    report = dict(mode='OFFLINE_SHARED_PIPELINE_REPLAY', cycles=cycles, matching_cycles=matching,
                  mismatching_cycles=cycles-matching, identical=cycles == matching and terminal_matches is not False,
                  terminal_recorded=terminal_recorded,terminal_matches=terminal_matches,
                  observed_seconds=(elapsed_end-elapsed_start), acknowledged_cycles=acknowledged,
                  feedback_events=events, first_mismatches=mismatches, automatic_inputs=False,
                  measurement='Recorded observations, model deliveries and input receipts through the shared pipeline. '
                              'No counterfactual game outcomes or live performance claim.')
    atomic_json(output or folder/'pipeline_replay_report.json', report)
    return report
