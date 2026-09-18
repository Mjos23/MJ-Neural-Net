"""Behavioral contracts for the provisional neural profile, through Elsa and Joanna."""
from copy import deepcopy
from pathlib import Path
from decimal import Decimal
import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from mj_neural_net import evaluate, example_request
from mj_neural_net import adapter


def run_source(body):
    sources = {'neural.bangel': (adapter.HERE/'source/neural.bangel').read_text(),
               'probe.bangel': 'bangel 1.0\nprogram probe\neffects pure\nimport mj.neural as neural\n' + body}
    raw = adapter.ElsaCompiler().compile_project(sources, entry_source_name='probe.bangel').to_bytes()
    return adapter.plain(adapter.execute_jp(raw)['outputs'])


def transition(key, previous, following, timestamp):
    return dict(id=key, previous=previous, following=following, observed_at=timestamp,
                available_at=timestamp+1, provenance='OBSERVED')


def update_request(following='derive'):
    request = example_request()
    request['history'] = [transition(f'h{i}', 'emit', 'let', 30+i*2) for i in range(3)]
    request['updates'] = [transition(f'u{i}', 'emit', following, 60+i*2) for i in range(8)]
    request['validation'] = [transition(f'v{i}', 'emit', 'derive', 100+i*2) for i in range(3)]
    return request


class NeuralContractTests(unittest.TestCase):
    def test_partial_cue_reconstructs_stored_pattern(self):
        request = example_request()
        result = evaluate(request)
        self.assertEqual(result['candidate'], 'memory-a')
        self.assertTrue(result['baseline_available'])
        self.assertTrue(result['match_available'])
        self.assertEqual(result['r0']['status'], 'PROVISIONAL')
        self.assertFalse(result['physical_actuation_allowed'])

    def test_relational_near_miss_is_held(self):
        request = example_request()
        request['query']['context'] = 'unfamiliar-context'
        result = evaluate(request)
        self.assertTrue(result['baseline_available'])
        self.assertFalse(result['match_available'])
        self.assertEqual(result['status'], 'RELATION_HOLD')

    def test_unknown_is_not_observed_zero(self):
        request = example_request()
        request['query']['cue'] = [None] * 9
        result = evaluate(request)
        self.assertEqual(result['known_features'], 0)
        self.assertFalse(result['baseline_available'])
        request['query']['cue'][0] = '0'
        self.assertEqual(evaluate(request)['known_features'], 1)

    def test_reconstruction_recovers_a_corrupted_known_bit(self):
        request = example_request()
        request['query']['cue'] = list(request['memories'][0]['pattern'])
        request['query']['cue'][0] = '-0.9'
        result = evaluate(request)
        self.assertEqual(result['candidate'], 'memory-a')
        self.assertGreater(Decimal(result['layers'][-1]['n1']), 0)

    def test_missing_context_holds_a_known_entity(self):
        request = example_request()
        request['query']['context'] = None
        result = evaluate(request)
        self.assertTrue(result['baseline_available'])
        self.assertFalse(result['match_available'])

    def test_aliasing_two_memories_is_ambiguous(self):
        request = example_request()
        duplicate = deepcopy(request['memories'][0])
        duplicate.update(id='ambiguous-memory', entity='robot-b')
        request['memories'].append(duplicate)
        result = evaluate(request)
        self.assertFalse(result['baseline_available'])
        self.assertIsNone(result['candidate'])

    def test_independent_deny_is_never_overridden(self):
        request = example_request()
        review = dict(profile='MJ-Neural-Review/0.1.0', request_sha256=adapter.digest(request), verdict='DENY')
        result = evaluate(request, review=review)
        self.assertTrue(result['match_available'])
        self.assertEqual(result['review_status'], 'DOWNSTREAM_DENY')
        self.assertFalse(result['advisory_ready'])

    def test_missing_review_remains_hold(self):
        result = evaluate(example_request())
        self.assertEqual(result['review_status'], 'DOWNSTREAM_HOLD')
        self.assertFalse(result['advisory_ready'])

    def test_pass_review_only_makes_advisory_available(self):
        request = example_request()
        review = dict(profile='MJ-Neural-Review/0.1.0', request_sha256=adapter.digest(request), verdict='PASS')
        result = evaluate(request, review=review, include_receipt=True)
        self.assertTrue(result['advisory_ready'])
        self.assertFalse(result['physical_actuation_allowed'])
        self.assertFalse(result['live_trading_allowed'])
        self.assertEqual(result['receipt']['authority_effect'], 'NONE')
        self.assertEqual(result['receipt']['physical_effect'], 'NONE')
        self.assertEqual(result['receipt']['traversal'][0]['stage'], 'R0')
        self.assertFalse(result['r0']['mapped_to_R0'])

    def test_improved_held_out_loss_selects_new_first_node(self):
        result = evaluate(update_request())
        self.assertTrue(result['kernel_review']['promoted'])
        self.assertLess(Decimal(result['kernel_review']['candidate_loss']), Decimal(result['kernel_review']['prior_loss']))
        self.assertEqual(result['placements'][0]['path']['nodes'][0], 3)

    def test_regressing_update_retains_previous_path(self):
        request = update_request('let')
        result = evaluate(request)
        self.assertFalse(result['kernel_review']['promoted'])
        self.assertEqual(result['placements'][0]['path']['nodes'][0], 1)

    def test_missing_validation_cannot_promote(self):
        request = update_request()
        request['validation'] = []
        result = evaluate(request)
        self.assertFalse(result['kernel_review']['promoted'])
        self.assertIsNone(result['kernel_review']['prior_loss'])
        self.assertEqual(result['placements'][0]['path']['nodes'][0], 1)


class MathematicsTests(unittest.TestCase):
    def test_maximum_profile_input_fits_the_existing_receipt_contract(self):
        request = example_request()
        for index in range(3):
            memory = deepcopy(request['memories'][0])
            memory.update(id=f'max-memory-{index}', entity=f'max-robot-{index}')
            old = memory['pattern'][index]
            memory['pattern'][index] = '-0.9' if old == '0.9' else '0.9'
            request['memories'].append(memory)
        for split, count, start in [('history', 32, 100), ('updates', 32, 200), ('validation', 16, 300)]:
            request[split] = [transition(f'{split}-{i}', 'emit', 'derive', start+i) for i in range(count)]
        result = evaluate(request)
        self.assertEqual(result['status'], 'ASSOCIATIVE_HOLD')
        self.assertLess(result['resources']['events_used'], 50000)
        self.assertTrue(result['kernel_review']['promoted'])

    def test_paper_prediction_gain_and_runtime_bounds(self):
        result = run_source('emit first = neural.prediction_gain(1)\n'
            'emit ninth = neural.prediction_gain(9)\n'
            'emit capped = neural.activation_step(0.0, 1.0, true)\n'
            'emit damped = neural.activation_step(0.0, 1.0, false)\n'
            'emit upper = neural.activation_step(0.9, 1.0, false)\n')
        self.assertEqual(Decimal(result['first']), Decimal('0.1'))
        self.assertLess(abs(Decimal(result['ninth']) - Decimal(256)/Decimal(65610)), Decimal('1e-28'))
        self.assertEqual(Decimal(result['capped']), Decimal('0.1'))
        self.assertEqual(Decimal(result['damped']), Decimal('0.3'))
        self.assertEqual(Decimal(result['upper']), Decimal('0.9'))

    def test_hebbian_weights_have_expected_values_and_no_self_connections(self):
        request = example_request()
        sources = adapter.sources_for(request)
        # Check the actual emitted weight values through public inference.
        result = evaluate(request)
        weights = result['weights']
        for i in range(1, 10):
            self.assertEqual(Decimal(weights[f'n{i}'][f'n{i}']), 0)
            for j in range(1, 10):
                self.assertEqual(weights[f'n{i}'][f'n{j}'], weights[f'n{j}'][f'n{i}'])
        self.assertEqual(Decimal(weights['n1']['n2']), Decimal('0.09'))
        self.assertEqual(Decimal(weights['n1']['n4']), Decimal('-0.09'))
        self.assertIn('neural.train(memories)', sources['main.bangel'])

    def test_every_layer_visits_all_nine_logical_nodes(self):
        result = evaluate(example_request())
        self.assertEqual(len(result['layers']), 3)
        for placement in result['placements']:
            nodes = placement['path']['nodes']
            self.assertEqual(sorted(nodes), list(range(1, 10)))
            for a, b in zip(nodes, nodes[1:]):
                self.assertEqual(abs((a-1)//3-(b-1)//3) + abs((a-1)%3-(b-1)%3), 1)
        for layer in result['layers']:
            self.assertTrue(all(abs(Decimal(value)) <= Decimal('0.9') for value in layer.values()))

    def test_jp_reexecution_is_deterministic(self):
        raw = adapter.ElsaCompiler().compile_project(adapter.sources_for(example_request()), entry_source_name='main.bangel').to_bytes()
        self.assertEqual(adapter.execute_jp(raw), adapter.execute_jp(raw))

    def test_invalid_ir_and_exhausted_budget_fail_closed(self):
        with self.assertRaises(Exception):
            adapter.execute_jp(b'{"profile":"JP-IR/1"}')
        raw = adapter.ElsaCompiler().compile_project(adapter.sources_for(example_request()), entry_source_name='main.bangel').to_bytes()
        with patch.dict(adapter.LIMITS, {'instruction_limit': 1}):
            with self.assertRaises(adapter.BangelDiagnostic):
                adapter.execute_jp(raw)


class BoundaryTests(unittest.TestCase):
    def test_producer_cannot_supply_review_fields_or_R0(self):
        for field, value in [('verdict', 'PASS'), ('r0', 'R0'), ('review', {}), ('paths', [])]:
            request = example_request()
            request[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                adapter.sources_for(request)

    def test_review_is_bound_to_exact_request(self):
        request = example_request()
        review = dict(profile='MJ-Neural-Review/0.1.0', request_sha256='0'*64, verdict='PASS')
        with self.assertRaises(ValueError):
            adapter.sources_for(request, review=review)

    def test_bad_numeric_data_and_unknown_tokens_reject(self):
        for value in (0.1, True, 'NaN', 'Infinity', '1e2', '0.91', '-0.91', '"\nemit injected = true'):
            request = example_request()
            request['query']['cue'][0] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                adapter.sources_for(request)
        request = example_request()
        request['query']['previous_command'] = 'R0'
        with self.assertRaises(ValueError):
            adapter.sources_for(request)

    def test_future_and_synthetic_evidence_reject(self):
        for split in ('memories', 'history', 'updates', 'validation'):
            for field, value in [('available_at', 1000), ('available_at', 1001), ('provenance', 'SYNTHETIC_NEAR_MISS')]:
                request = update_request()
                request[split][0][field] = value
                with self.subTest(split=split, field=field, value=value), self.assertRaises(ValueError):
                    adapter.sources_for(request)

    def test_train_validation_overlap_and_duplicate_identity_reject(self):
        request = update_request()
        request['validation'][0]['id'] = request['history'][0]['id']
        with self.assertRaises(ValueError):
            adapter.sources_for(request)
        request = update_request()
        request['validation'][0]['observed_at'] = 1
        with self.assertRaises(ValueError):
            adapter.sources_for(request)

    def test_first_observation_is_not_moved_forward(self):
        request = example_request()
        request['as_of'] += 1
        with self.assertRaises(ValueError):
            adapter.sources_for(request)

    def test_oversized_and_empty_inputs_reject(self):
        for field, values in [('memories', []), ('history', [{}]*33), ('updates', [{}]*33), ('validation', [{}]*17)]:
            request = example_request()
            request[field] = values
            with self.subTest(field=field), self.assertRaises(ValueError):
                adapter.sources_for(request)
        for size in (8, 10):
            request = example_request()
            request['query']['cue'] = ['0']*size
            with self.assertRaises(ValueError):
                adapter.sources_for(request)

    def test_json_duplicate_fields_and_nonfinite_reject(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'request.json'
            for raw in ('{"x":1,"x":2}', '{"x":NaN}', ' '*65537):
                path.write_text(raw)
                with self.subTest(raw=raw[:30]), self.assertRaises(ValueError):
                    adapter.read_json(path)

    def test_cli_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'result.json'
            output.write_text('preserve')
            process = subprocess.run([sys.executable, '-B', '-m', 'mj_neural_net', 'run',
                str(adapter.HERE/'examples/partial-cue.json'), '--output', str(output)], capture_output=True)
            self.assertEqual(process.returncode, 2)
            self.assertEqual(output.read_text(), 'preserve')

    def test_field_order_does_not_change_jp(self):
        request = example_request()
        reordered = dict(reversed(list(request.items())))
        self.assertEqual(adapter.sources_for(request), adapter.sources_for(reordered))


if __name__ == '__main__':
    unittest.main()
