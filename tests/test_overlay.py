"""Source coverage, native diffusion and whole-request composition boundaries."""
from copy import deepcopy
from decimal import Decimal, localcontext
import hashlib
import io
import json
import unittest

from mj_neural_net import adapter, catalog, compose, example_envelope, overlay
from mj_neural_net import integration
from mj_neural_net.api import create_app


def review(value, verdict='PASS'):
    return {'profile': integration.REVIEW_PROFILE, 'request_sha256': adapter.digest(value), 'verdict': verdict}


def paired_request():
    cue = ['0.9'] * 9
    def events(offset):
        return [{'actor': 'sensor', 'step': i, 'position': [str(100+i*60+offset), str(150+i*20)],
                 'command': token} for i, token in enumerate(['measure', 'derive', 'measure', 'derive'])]
    memories = [{'id': 'memory-' + name, 'namespace': 'robotics.lab', 'pattern': cue,
        'entity': 'robot', 'relation': 'inspection', 'context': 'bench',
        'observed_at': 10, 'available_at': 11, 'provenance': 'OBSERVED',
        'receipt_ref': 'fixture-' + name, 'selection_id': None, 'trace': events(offset)}
        for name, offset in [('a', 0), ('b', 300)]]
    query = {'id': 'query-1', 'observed_at': 100, 'cue': cue, 'entity': 'robot',
             'relation': 'inspection', 'context': 'bench', 'previous_command': 'measure', 'events': events(0)}
    memory = {'profile': 'MJ-Memory-Recall/0.1.0', 'namespace': 'robotics.lab', 'as_of': 100,
        'frame': {'id': 'local-grid', 'x_min': '0', 'y_min': '0', 'width': '800', 'height': '600'},
        'memories': memories, 'query': query}
    neural = {'profile': adapter.PROFILE, 'as_of': 100,
        'memories': [{key: value for key, value in m.items() if key not in
                      ('namespace', 'receipt_ref', 'selection_id', 'trace')} for m in memories],
        'query': {key: value for key, value in query.items() if key != 'events'},
        'history': [], 'updates': [], 'validation': []}
    return {'profile': integration.PROFILE, 'neural': deepcopy(neural), 'memory_recall': memory}


def invoke(app, value, path='/v1/compose', method='POST'):
    raw = json.dumps(value).encode()
    environ = {'REQUEST_METHOD': method, 'PATH_INFO': path, 'QUERY_STRING': '',
        'CONTENT_TYPE': 'application/json', 'CONTENT_LENGTH': str(len(raw)), 'wsgi.input': io.BytesIO(raw)}
    status = []
    encoded = b''.join(app(environ, lambda text, headers: status.append(text)))
    return int(status[0].split()[0]), json.loads(encoded)


class DiagramSourceTests(unittest.TestCase):
    def test_all_five_templates_and_every_source_occurrence_are_preserved(self):
        data = catalog()
        diagrams = data['diagrams']
        self.assertEqual([t['source_count'] for t in diagrams['templates']], [19, 24, 17, 27, 13])
        self.assertEqual([len(t['edges']) for t in diagrams['templates']], [3, 8, 5, 3, 2])
        self.assertEqual(len(data['teal_mappings']['entries']), 500)
        self.assertEqual({o['yellow_pdf_page'] for o in diagrams['occurrences']}, set(range(1, 101)))
        for occurrence in diagrams['occurrences']:
            seed = int(hashlib.sha256(occurrence['yellow_id'].encode()).hexdigest()[:8], 16)
            self.assertEqual(occurrence['assignment_mode'], seed % 5)
            self.assertEqual(int(occurrence['teal_id'][3:]), occurrence['yellow_pdf_page'] + 300)
        grid = diagrams['templates'][1]
        self.assertEqual([p['normalized_grid_coordinates'] for p in grid['nodes']],
                         [[0,0],[1,0],[2,0],[0,1],[1,1],[2,1],[0,2],[1,2],[2,2]])
        self.assertEqual([(e['from'].rsplit(':',1)[1], e['to'].rsplit(':',1)[1]) for e in grid['edges']],
                         [('N'+str(i), 'N'+str(i+1)) for i in range(1,9)])
        for template in diagrams['templates']:
            self.assertFalse(template['approved_sequence'])
            self.assertTrue(all(not e['directed'] for e in template['edges']))
        copied = catalog()
        copied['diagrams']['templates'].clear()
        self.assertEqual(len(catalog()['diagrams']['templates']), 5)

    def test_conflicting_picture_and_text_are_separate(self):
        item = next(o for o in catalog()['diagrams']['occurrences'] if o['yellow_id'] == 'YP-093')
        self.assertEqual(item['assignment_mode'], 3)
        self.assertEqual(item['semantic_path_text_separate_from_diagram'], 'R -> R0 -> R1 -> N1 -> R2')
        self.assertEqual(item['semantic_alignment_to_template'], 'NOT_ASSERTED_HASH_ASSIGNED')


class NativeOverlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.request = example_envelope()
        cls.result = overlay(cls.request, include_receipt=True)

    def test_every_layer_executes_and_combines_with_bounded_diffusion(self):
        result = self.result
        self.assertEqual(len(result['layers']), 5)
        self.assertEqual(result['projected_nodes'][1], list(range(1,10)))
        self.assertEqual([len(e) for e in result['projected_edges']], [3,8,5,3,2])
        self.assertEqual(result['receipt']['status'], 'PASS')
        self.assertGreater(result['resources']['events_used'], 0)
        self.assertEqual(result['review_status'], 'DOWNSTREAM_HOLD')
        self.assertNotEqual(result['forecast'], result['combined'])
        with localcontext() as context:
            context.prec = 45
            for key, start in result['initial'].items():
                initial = Decimal(start)
                mean = sum(Decimal(layer[key]) for layer in result['layers']) / 5
                delta = max(Decimal('-0.1'), min(Decimal('0.1'), (mean-initial)/3))
                self.assertLess(abs(Decimal(result['combined'][key]) - (initial+delta)), Decimal('1e-30'))
                for layer in result['layers']:
                    self.assertLessEqual(abs(Decimal(layer[key])-initial), Decimal('0.100000000000000000000000000000001'))
        self.assertFalse(result['approved_traversal_claim'])

    def test_absent_command_evidence_and_equal_scores_hold(self):
        value = example_envelope()
        value['neural']['history'] = []
        result = overlay(value, review=review(value))
        self.assertFalse(result['placement']['available'])
        self.assertIsNone(result['placement']['node'])
        self.assertEqual(result['placement']['margin'], '0')
        self.assertFalse(result['advisory_ready'])

    def test_kernel_improvement_requires_independent_whole_envelope_review(self):
        value = example_envelope()
        value['neural']['as_of'] = value['neural']['query']['observed_at'] = 1000
        def records(prefix, count, start, following):
            return [{'id': prefix+str(i), 'previous':'emit', 'following':following,
                     'observed_at':start+i, 'available_at':start+i, 'provenance':'OBSERVED'} for i in range(count)]
        value['neural']['history'] = records('h', 32, 100, 'measure')
        value['neural']['updates'] = records('u', 32, 200, 'derive')
        value['neural']['validation'] = records('v', 16, 300, 'derive')
        value['neural']['query']['previous_command'] = 'emit'
        held = overlay(value)
        admitted = overlay(value, review=review(value))
        self.assertTrue(held['kernel_review']['promoted'])
        self.assertFalse(held['kernel_update_admitted'])
        self.assertTrue(admitted['kernel_update_admitted'])
        self.assertNotEqual(held['forecast'], admitted['forecast'])
        self.assertLess(admitted['resources']['events_used'], adapter.LIMITS['event_limit'])

    def test_native_conflict_and_any_downstream_deny_dominate(self):
        sources = {'neural.bangel': (adapter.HERE/'source/neural.bangel').read_text(),
                   'overlay.bangel': (adapter.HERE/'source/overlay.bangel').read_text(),
                   'main.bangel': 'bangel 1.0\nprogram gate_check\neffects pure\nimport mj.neural_overlay as overlay\n'
                   'emit conflict = overlay.decide("a", true, "b", true, true, "PASS", false, false)\n'
                   'emit denied = overlay.decide("a", true, "a", true, true, "PASS", false, true)\n'}
        raw = adapter.ElsaCompiler().compile_project(sources, entry_source_name='main.bangel').to_bytes()
        values = adapter.plain(adapter.execute_jp(raw)['outputs'])
        self.assertTrue(values['conflict']['conflict'])
        self.assertFalse(values['conflict']['matched'])
        self.assertEqual(values['conflict']['review_status'], 'DOWNSTREAM_HOLD')
        self.assertEqual(values['denied']['review_status'], 'DOWNSTREAM_DENY')


class CompositionTests(unittest.TestCase):
    def test_native_recall_recovers_baseline_tie_without_fabricating_context(self):
        value = paired_request()
        result = compose(value, review=review(value))
        self.assertFalse(result['baseline']['match_available'])
        self.assertTrue(result['memory_recall']['match_available'])
        self.assertEqual(result['candidate'], 'memory-a')
        self.assertTrue(result['advisory_ready'])
        self.assertTrue(result['overlay']['recall_forecast_admitted'])
        self.assertEqual(result['overlay']['placement']['node'], 3)
        self.assertEqual(result['memory_recall']['request_sha256'], adapter.digest(value['memory_recall']))
        self.assertFalse(result['r0']['mapped_to_R0'])

    def test_mismatched_child_ids_cues_context_and_extra_fields_reject_before_execution(self):
        original = paired_request()
        mutations = [('cue', ['-0.9']*9), ('context','different'), ('id','different-query'), ('previous_command','emit')]
        for field, replacement in mutations:
            value = deepcopy(original)
            value['memory_recall']['query'][field] = replacement
            with self.subTest(field=field), self.assertRaises(ValueError):
                integration.validate(value)
        value = deepcopy(original)
        value['memory_recall']['memories'][0]['id'] = 'other-memory'
        with self.assertRaises(ValueError):
            integration.validate(value)
        value = deepcopy(original)
        value['memory_recall']['memories'][0]['pattern'][0] = '-0.9'
        with self.assertRaises(ValueError):
            integration.validate(value)
        value = deepcopy(original)
        value['review'] = review(value)
        with self.assertRaises(ValueError):
            integration.validate(value)

    def test_inner_request_review_cannot_authorize_whole_envelope(self):
        value = paired_request()
        child_review = {'profile': integration.REVIEW_PROFILE,
                        'request_sha256':adapter.digest(value['neural']), 'verdict':'PASS'}
        with self.assertRaises(ValueError):
            compose(value, review=child_review)
        whole = review(value)
        value['memory_recall']['query']['events'][0]['position'][0] = '120'
        with self.assertRaises(ValueError):
            compose(value, review=whole)

    def test_http_routes_bind_review_and_hold_by_default(self):
        value = example_envelope()
        status, result = invoke(create_app(), value, '/v1/overlay')
        self.assertEqual(status, 200)
        self.assertEqual(result['review_status'], 'DOWNSTREAM_HOLD')
        def deny(snapshot):
            record = review(snapshot, 'DENY')
            snapshot['neural']['query']['context'] = 'mutated'
            return record
        status, result = invoke(create_app(review_provider=deny), value)
        self.assertEqual(status, 200)
        self.assertEqual(result['review_status'], 'DOWNSTREAM_DENY')
        self.assertEqual(result['request_sha256'], adapter.digest(value))
        status, result = invoke(create_app(), {}, '/v1/catalog', 'GET')
        self.assertEqual(status, 200)
        self.assertEqual(len(result['diagrams']['occurrences']), 100)
        status, _ = invoke(create_app(review_provider=lambda _:review(value)), value['neural'], '/v1/compose')
        self.assertEqual(status, 400)


if __name__ == '__main__':
    unittest.main()
