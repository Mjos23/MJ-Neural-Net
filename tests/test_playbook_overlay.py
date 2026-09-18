"""Independent source-curve projection, layer arithmetic and review gating."""
from copy import deepcopy
from decimal import Decimal, localcontext
from fractions import Fraction
import unittest

from mj_memory_recall.engine import catalog_index, path_command, selection
from mj_neural_net import overlay
from test_overlay import paired_request, review


OFFENSE = 'dlxs:offense:inside-zone:base-bangel'
DEFENSE = 'dlxs:defense:cover-three:base-bangel'


def playbook_request(identity=OFFENSE):
    value = paired_request()
    item = selection(identity)
    memory = value['memory_recall']
    memory['frame'] = deepcopy(catalog_index()['frame'])
    memory['memories'] = [memory['memories'][0]]
    memory['memories'][0].update(selection_id=identity, trace=[])
    memory['query']['events'] = [{'actor': path['actor'], 'step': 8,
        'position': [str(x) for x in path['end']], 'command': path_command(path['kind'])}
        for path in item['paths'][:3]]
    value['neural']['memories'] = [value['neural']['memories'][0]]
    value['neural']['history'] = [{'id': 'history-'+str(i), 'previous': 'measure',
        'following': 'derive', 'observed_at': 20+i, 'available_at': 20+i,
        'provenance': 'OBSERVED'} for i in range(3)]
    return value


def expected_nodes(path, frame):
    # Exact independent quadratic evaluation and half-open thirds.
    xmin, ymin, width, height = (Fraction(frame[k]) for k in ('x_min', 'y_min', 'width', 'height'))
    result = []
    for step in range(9):
        t = Fraction(step, 8)
        u = 1-t
        x, y = [u*u*a + 2*u*t*b + t*t*c for a,b,c in zip(path['start'], path['control'], path['end'])]
        column = int(x >= xmin+width/3) + int(x >= xmin+2*width/3)
        row = 2-int(y >= ymin+height/3)-int(y >= ymin+2*height/3)
        result.append(1+column+3*row)
    return result


class PlaybookSpatialOverlay(unittest.TestCase):
    def test_offense_and_defense_source_paths_drive_native_layer(self):
        for identity in (OFFENSE, DEFENSE):
            with self.subTest(selection=identity):
                value = playbook_request(identity)
                result = overlay(value, review=review(value))
                layer = result['playbook_layer']
                self.assertTrue(layer['admitted'])
                self.assertEqual(layer['selection_id'], identity)
                self.assertEqual(len(layer['paths']), 11)
                self.assertEqual(layer['frame'], value['memory_recall']['frame'])
                self.assertNotEqual(result['combined'], result['teal_combined'])
                self.assertEqual(result['review_status'], 'MATCH_FOR_REVIEW')
                self.assertIn('recall.bangel', result['source_sha256'])
                self.assertIn('playbook.bangel', result['source_sha256'])
                self.assertFalse(result['recall_forecast_admitted'])
                for path, original in zip(layer['paths'], selection(identity)['paths']):
                    self.assertEqual({k:v for k,v in path.items() if k != 'nodes'}, original)
                    self.assertEqual(path['nodes'], expected_nodes(original, layer['frame']))
                edges = [(a,b) for path in layer['paths'] for a,b in zip(path['nodes'], path['nodes'][1:])]
                self.assertEqual(len(edges), 88)
                self.assertTrue(any(a == b for a,b in edges))
                with localcontext() as context:
                    context.prec = 45
                    initial = [Decimal(result['initial']['n'+str(i)]) for i in range(1,10)]
                    for index in range(1,10):
                        neighbors = ([initial[b-1] for a,b in edges if a == index] +
                                     [initial[a-1] for a,b in edges if b == index])
                        delta = ((sum(neighbors)/len(neighbors)-initial[index-1])/3) if neighbors else Decimal(0)
                        expected = initial[index-1]+max(Decimal('-0.1'), min(Decimal('0.1'), delta))
                        key = 'n'+str(index)
                        self.assertLess(abs(Decimal(layer['scores'][key])-expected), Decimal('1e-30'))
                        mean = (sum(Decimal(x[key]) for x in result['layers']) + expected) / 6
                        combined = initial[index-1]+max(Decimal('-0.1'), min(Decimal('0.1'), (mean-initial[index-1])/3))
                        self.assertLess(abs(Decimal(result['combined'][key])-combined), Decimal('1e-30'))

    def test_missing_or_denied_review_cannot_admit_spatial_placement(self):
        value = playbook_request()
        for independent in (None, review(value, 'DENY')):
            result = overlay(value, review=independent)
            self.assertFalse(result['playbook_layer']['admitted'])
            self.assertEqual(result['combined'], result['teal_combined'])
            self.assertFalse(result['advisory_ready'])
            self.assertEqual(result['review_status'], 'DOWNSTREAM_HOLD' if independent is None else 'DOWNSTREAM_DENY')

    def test_ambiguous_candidate_does_not_create_a_playbook_layer(self):
        value = playbook_request()
        for key in ('neural', 'memory_recall'):
            other = deepcopy(value[key]['memories'][0])
            other['id'] = 'memory-b'
            if key == 'memory_recall':
                other['receipt_ref'] = 'other-fixture'
            value[key]['memories'].append(other)
        result = overlay(value, review=review(value))
        self.assertIsNone(result['playbook_layer'])
        self.assertEqual(result['combined'], result['teal_combined'])


if __name__ == '__main__':
    unittest.main()
