"""Installed package and public API contracts for the cloned neural baseline."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import unittest

from mj_neural_net import NAME, evaluate, example_request
from mj_neural_net import adapter
from mj_neural_net.api import create_app

FIXTURES = Path(__file__).parent / 'fixtures'


def invoke(app, method='POST', path='/v1/evaluate', body=None, **overrides):
    raw = json.dumps(example_request()).encode() if body is None else body
    environ = {'REQUEST_METHOD': method, 'PATH_INFO': path, 'QUERY_STRING': '',
               'CONTENT_TYPE': 'application/json', 'CONTENT_LENGTH': str(len(raw)),
               'wsgi.input': io.BytesIO(raw)}
    environ.update(overrides)
    response = {}

    def start_response(status, headers):
        response['status'] = int(status.split()[0])
        response['headers'] = dict(headers)

    encoded = b''.join(app(environ, start_response))
    response['body'] = json.loads(encoded)
    return response


class PackageContractTests(unittest.TestCase):
    def test_clone_matches_recorded_baseline_result(self):
        expected = json.loads((FIXTURES / 'baseline-partial-result.json').read_text())
        actual = evaluate(example_request())
        self.assertEqual(actual.pop('name'), 'MJ Neural Net')
        identity = actual.pop('implementation')
        self.assertEqual(identity['mode'], 'CLONED_BASELINE')
        expected.pop('name')
        self.assertEqual(actual, expected)

    def test_neural_source_is_the_attributed_baseline(self):
        raw = (adapter.HERE / 'source/neural.bangel').read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), adapter.BASELINE_SOURCE_SHA256)
        self.assertEqual(NAME, 'MJ Neural Net')
        self.assertNotIn('sys.path', (adapter.HERE / 'adapter.py').read_text())

    def test_packaged_resources_are_available(self):
        self.assertEqual(len(adapter.load_paths()['paths']), 5)
        self.assertEqual(len(example_request()['query']['cue']), 9)
        self.assertTrue((adapter.HERE / 'docs/r0-0.1.0.md').is_file())
        self.assertEqual(json.loads((adapter.HERE / 'PLUGIN.json').read_text())['name'], NAME)


class HTTPContractTests(unittest.TestCase):
    def test_profile_requires_no_execution_or_review(self):
        def unavailable(_):
            raise AssertionError('Profile must not call the reviewer')
        response = invoke(create_app(review_provider=unavailable), method='GET', path='/v1/profile')
        self.assertEqual(response['status'], 200)
        self.assertEqual(response['body']['name'], 'MJ Neural Net')
        self.assertEqual(response['body']['request_profile'], adapter.PROFILE)
        self.assertFalse(response['body']['r0']['mapped_to_R0'])
        self.assertEqual(response['headers']['Cache-Control'], 'no-store')

    def test_http_default_remains_held_and_has_no_receipt(self):
        response = invoke(create_app())
        self.assertEqual(response['status'], 200)
        self.assertEqual(response['body']['candidate'], 'memory-a')
        self.assertEqual(response['body']['review_status'], 'DOWNSTREAM_HOLD')
        self.assertFalse(response['body']['advisory_ready'])
        self.assertNotIn('receipt', response['body'])

    def test_server_review_cannot_modify_request_or_overturn_denial(self):
        request = example_request()
        def deny(snapshot):
            review = {'profile': 'MJ-Neural-Review/0.1.0',
                      'request_sha256': adapter.digest(snapshot), 'verdict': 'DENY'}
            snapshot['query']['context'] = 'mutated-by-reviewer'
            return review
        response = invoke(create_app(review_provider=deny), body=json.dumps(request).encode())
        self.assertEqual(response['status'], 200)
        self.assertTrue(response['body']['match_available'])
        self.assertEqual(response['body']['request_sha256'], adapter.digest(request))
        self.assertEqual(response['body']['review_status'], 'DOWNSTREAM_DENY')
        self.assertFalse(response['body']['physical_actuation_allowed'])

    def test_json_boundaries_reject_before_review(self):
        def must_not_run(_):
            raise AssertionError('Invalid input reached reviewer')
        app = create_app(review_provider=must_not_run)
        invalid = [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}', b'\xff',
                   b'[' * 1000 + b']' * 1000, b'null', b'[]']
        for raw in invalid:
            with self.subTest(raw=raw[:30]):
                self.assertEqual(invoke(app, body=raw)['status'], 400)
        request = example_request()
        for field, value in [('review', {'verdict': 'PASS'}), ('source', 'emit allowed = true'),
                             ('paths', []), ('r0', 'R0'), ('include_receipt', True)]:
            changed = deepcopy(request)
            changed[field] = value
            with self.subTest(field=field):
                self.assertEqual(invoke(app, body=json.dumps(changed).encode())['status'], 400)

    def test_http_framing_and_routing_are_bounded(self):
        app = create_app()
        cases = [({'CONTENT_LENGTH': ''}, 411), ({'CONTENT_LENGTH': '-1'}, 400),
                 ({'CONTENT_LENGTH': '1' * 1000}, 413),
                 ({'CONTENT_LENGTH': '65537'}, 413),
                 ({'CONTENT_LENGTH': '10', 'body': b'{}'}, 400),
                 ({'CONTENT_TYPE': 'text/plain'}, 415),
                 ({'CONTENT_TYPE': 'application/json; charset=latin-1'}, 415),
                 ({'QUERY_STRING': 'review=PASS'}, 400),
                 ({'method': 'GET'}, 405), ({'path': '/other'}, 404)]
        for arguments, status in cases:
            with self.subTest(arguments=arguments):
                self.assertEqual(invoke(app, **arguments)['status'], status)

    def test_reviewer_failure_or_bad_binding_fails_closed(self):
        def unavailable(_):
            raise RuntimeError('private reviewer error')
        def wrong_binding(_):
            return {'profile': 'MJ-Neural-Review/0.1.0', 'request_sha256': '0' * 64, 'verdict': 'PASS'}
        for callback in (unavailable, wrong_binding):
            response = invoke(create_app(review_provider=callback))
            self.assertEqual(response['status'], 503)
            self.assertEqual(response['body']['error']['code'], 'REVIEW_UNAVAILABLE')
            self.assertNotIn('private reviewer error', json.dumps(response['body']))


if __name__ == '__main__':
    unittest.main()
