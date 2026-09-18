"""Typed transport and source binding. Neural/model arithmetic lives in .bangel."""
from __future__ import annotations

import argparse
from copy import deepcopy
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import sys
import time

HERE = Path(__file__).resolve().parent
from bangel.compiler import ElsaCompiler
from bangel.diagnostics import BangelDiagnostic
from bangel.ir import JPIR
from bangel.runtime import JoannaRuntime
from bangel.receipts import verify_receipt

NAME = 'MJ Neural Net'
VERSION = '0.1.0'
BASELINE_NAME = 'MJ-Bangel-Neural-Network-Plugin'
BASELINE_SOURCE_SHA256 = '39b91c4ac89a15eb6267ae38436d186a1ffec370205637880cc60af9adcc0d9e'
PROFILE = 'MJ-Bangel-Neural/0.1.0'
COMMANDS = ('let', 'measure', 'derive', 'require', 'if', 'for', 'match', 'emit', 'return')
LIMITS = dict(instruction_limit=50000, event_limit=50000, function_depth=64,
              evaluation_limit=1000000, allocation_limit=16000000,
              output_limit=1048576, receipt_limit=16777216)
TOKEN = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.:/-]{0,95}\Z')
REAL = re.compile(r'-?(?:0|[1-9][0-9]*)(?:\.[0-9]{1,12})?\Z')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def shape(value, keys, label):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError(label + ': fields do not match the closed profile')


def identity(value, label):
    if type(value) is not str or not TOKEN.fullmatch(value):
        raise ValueError(label + ': a bounded identifier is required')
    return value


def tick(value, label):
    if type(value) is not int or not 0 <= value <= 9007199254740991:
        raise ValueError(label + ': a nonnegative exact integer is required')
    return value


def real(value, label):
    if type(value) is not str or len(value) > 20 or not REAL.fullmatch(value):
        raise ValueError(label + ': a finite decimal string is required')
    if not Decimal('-0.9') <= Decimal(value) <= Decimal('0.9'):
        raise ValueError(label + ': outside [-0.9, 0.9]')
    return value if '.' in value else value + '.0'


def vector(value, maximum, label, minimum=0):
    if type(value) is not list or not minimum <= len(value) <= maximum:
        raise ValueError(label + ': invalid bounded list')


def command(value):
    if type(value) is not str or value not in COMMANDS:
        raise ValueError('Unknown command token; arbitrary source is not accepted')
    return COMMANDS.index(value) + 1


def validate(request):
    shape(request, ('profile', 'as_of', 'memories', 'history', 'updates', 'validation', 'query'), 'request')
    if request['profile'] != PROFILE:
        raise ValueError('Unsupported neural profile')
    as_of = tick(request['as_of'], 'as_of')
    shape(request['query'], ('id', 'observed_at', 'cue', 'entity', 'relation', 'context', 'previous_command'), 'query')
    query = request['query']
    identity(query['id'], 'query.id')
    observed_at = tick(query['observed_at'], 'query.observed_at')
    if observed_at != as_of:
        raise ValueError('as_of must be the first query observation; later data cannot backfill it')
    vector(query['cue'], 9, 'query.cue', 9)
    for value in query['cue']:
        if value is not None:
            real(value, 'cue')
    for field in ('entity', 'relation', 'context'):
        if query[field] is not None:
            identity(query[field], 'query.' + field)
    command(query['previous_command'])
    ids = {query['id']}

    def record(item):
        key = identity(item['id'], 'record.id')
        if key in ids:
            raise ValueError('Duplicate identity or train/validation/query leakage')
        ids.add(key)
        start = tick(item['observed_at'], 'observed_at')
        end = tick(item['available_at'], 'available_at')
        if not start <= end < as_of:
            raise ValueError('Observation unavailable at first predictive observation')
        if item['provenance'] != 'OBSERVED':
            raise ValueError('Synthetic evidence cannot train or validate the live model')
        return end

    vector(request['memories'], 4, 'memories', 1)
    for item in request['memories']:
        shape(item, ('id', 'pattern', 'entity', 'relation', 'context', 'observed_at', 'available_at', 'provenance'), 'memory')
        record(item)
        vector(item['pattern'], 9, 'memory.pattern', 9)
        for value in item['pattern']:
            real(value, 'memory.pattern')
            if Decimal(value) not in (Decimal('-0.9'), Decimal('0.9')):
                raise ValueError('Stored patterns use bipolar +/-0.9 features')
        for field in ('entity', 'relation', 'context'):
            identity(item[field], 'memory.' + field)
    times = {}
    for split, maximum in (('history', 32), ('updates', 32), ('validation', 16)):
        vector(request[split], maximum, split)
        times[split] = []
        for item in request[split]:
            shape(item, ('id', 'previous', 'following', 'observed_at', 'available_at', 'provenance'), split)
            times[split].append(record(item))
            command(item['previous'])
            command(item['following'])
    trained = times['history'] + times['updates']
    if request['validation'] and trained:
        if min(item['observed_at'] for item in request['validation']) <= max(trained):
            raise ValueError('Validation must follow every training/update availability time')
    if times['updates'] and times['history']:
        if min(item['observed_at'] for item in request['updates']) <= max(times['history']):
            raise ValueError('Updates must follow the prior training history')
    return request


def load_paths():
    catalog = json.loads((HERE / 'PATHWAYS.json').read_text())
    shape(catalog, ('profile', 'status', 'references', 'paths'), 'pathways')
    if catalog['profile'] != 'MJ-Neural-Pathways/0.1.0' or catalog['status'] != 'SIMULATION_REFERENCE':
        raise ValueError('Unrecognized pathway catalog')
    vector(catalog['paths'], 5, 'paths', 1)
    seen = set()
    for path in catalog['paths']:
        shape(path, ('id', 'nodes'), 'path')
        identity(path['id'], 'path.id')
        if path['id'] in seen:
            raise ValueError('Duplicate path identity')
        seen.add(path['id'])
        nodes = path['nodes']
        if (type(nodes) is not list or len(nodes) != 9 or
                any(type(n) is not int for n in nodes) or sorted(nodes) != list(range(1, 10))):
            raise ValueError('Path must visit each of the nine nodes once')
        for a, b in zip(nodes, nodes[1:]):
            if abs((a-1)//3 - (b-1)//3) + abs((a-1)%3 - (b-1)%3) != 1:
                raise ValueError('Path breaks the declared 3x3 adjacency')
    return catalog


def nodes(values):
    return 'neural.Nodes(' + ', '.join('n' + str(i+1) + ' = ' + real(x, 'node') for i, x in enumerate(values)) + ')'


def text_result(value):
    if value is None:
        return 'Result.Err(Failure(code = "MISSING_RELATION", message = "Unavailable relational evidence"))'
    return 'Result.Ok(' + json.dumps(value) + ')'


def transitions(items):
    return '[' + ', '.join('neural.Transition(previous = ' + str(command(x['previous'])) +
        ', following = ' + str(command(x['following'])) + ')' for x in items) + ']'


def review_binding(request, review):
    if review is None:
        return False, 'HOLD'
    shape(review, ('profile', 'request_sha256', 'verdict'), 'review')
    if (review['profile'] != 'MJ-Neural-Review/0.1.0' or review['request_sha256'] != digest(request)
            or review['verdict'] not in ('PASS', 'HOLD', 'DENY')):
        raise ValueError('Independent review is invalid or belongs to another request')
    return True, review['verdict']


def sources_for(request, *, review=None):
    validate(request)
    reviewed, verdict = review_binding(request, review)
    catalog = load_paths()
    query = request['query']
    memories = []
    for item in request['memories']:
        fields = ['id = ' + json.dumps(item['id']), 'pattern = ' + nodes(item['pattern'])]
        fields += [field + ' = ' + json.dumps(item[field]) for field in ('entity', 'relation', 'context')]
        memories.append('neural.Memory(' + ', '.join(fields) + ')')
    features = [('neural.Feature.Missing(reason = "Unobserved cue")' if x is None else
                 'neural.Feature.Known(value = ' + real(x, 'cue') + ')') for x in query['cue']]
    paths = ['neural.Path(id = ' + json.dumps(x['id']) + ', nodes = ' + json.dumps(x['nodes']) + ')'
             for x in catalog['paths']]
    lines = ['bangel 1.0', 'program mj_bangel_neural', 'effects pure', 'import mj.neural as neural',
        'let memories: Vector[neural.Memory] = [' + ', '.join(memories) + ']',
        'let cue: Vector[neural.Feature; 9] = [' + ', '.join(features) + ']',
        'let history: Vector[neural.Transition] = ' + transitions(request['history']),
        'let candidate_history: Vector[neural.Transition] = ' + transitions(request['history'] + request['updates']),
        'let validation: Vector[neural.Transition] = ' + transitions(request['validation']),
        'let paths: Vector[neural.Path] = [' + ', '.join(paths) + ']',
        'let relation_entity: Result[Text] = ' + text_result(query['entity']),
        'let relation_kind: Result[Text] = ' + text_result(query['relation']),
        'let relation_context: Result[Text] = ' + text_result(query['context']),
        'let prior_model = neural.train_commands(history)',
        'let candidate_model = neural.candidate_commands(prior_model, candidate_history, ' + str(bool(request['updates'])).lower() + ')',
        'let review = neural.review_update(prior_model, candidate_model, validation, ' + str(bool(request['updates'])).lower() + ')',
        'let weights = neural.train(memories)',
        'let known = neural.known_count(cue)',
        'let initial = neural.initialize(cue)',
        'var selected_model: neural.Weights = prior_model',
        'var selected_history: Vector[neural.Transition] = history',
        'if review.promoted', '  set selected_history = candidate_history', '  set selected_model = candidate_model', 'end',
        'var p1: neural.Placement = neural.place(selected_model, selected_history, ' + str(command(query['previous_command'])) + ', paths)',
        'var p2: neural.Placement = neural.place(selected_model, selected_history, neural.last_node(p1.path), paths)',
        'var p3: neural.Placement = neural.place(selected_model, selected_history, neural.last_node(p2.path), paths)',
        'var layer1: neural.Nodes = neural.layer(weights, initial, p1.path, true)',
        'var layer2: neural.Nodes = neural.layer(weights, layer1, p2.path, false)',
        'var layer3: neural.Nodes = neural.layer(weights, layer2, p3.path, false)',
        'var candidate: neural.Retrieval = neural.retrieve(memories, layer3, known)',
        'var matched: Bool = neural.relation_matches(candidate, relation_entity, relation_kind, relation_context)',
        'emit request_sha256 = ' + json.dumps(digest(request)),
        'emit pathway_sha256 = ' + json.dumps(digest(catalog)),
        'emit marker = "r0"', 'emit marker_status = "PROVISIONAL"',
        'emit marker_profile = "MJ-Neural-r0/0.1.0"',
        'emit known_features = known', 'emit original_cue = cue', 'emit initial_activation = initial',
        'emit weights = weights', 'emit layers = [layer1, layer2, layer3]',
        'emit placements = [p1, p2, p3]', 'emit kernel_review = review',
        'emit retrieval = candidate', 'emit match_available = matched',
        'emit review_status = neural.downstream_status(matched, ' + str(reviewed).lower() + ', ' + json.dumps(verdict) + ')',
        'emit physical_actuation_allowed = false']
    return {'neural.bangel': (HERE / 'source/neural.bangel').read_text(), 'main.bangel': '\n'.join(lines) + '\n'}


def execute_jp(raw):
    ir = JPIR.from_bytes(raw)
    ir.verify()
    deadline = time.monotonic() + 45
    receipt = JoannaRuntime(**LIMITS).execute(ir.to_dict(), cancelled=lambda: time.monotonic() > deadline).to_dict()
    verify_receipt(receipt)
    if receipt['status'] != 'PASS':
        raise ValueError('Neural execution did not pass: ' + str(receipt.get('hold_code')))
    if receipt['authority_effect'] != 'NONE' or receipt['physical_effect'] != 'NONE':
        raise ValueError('Unexpected authority or physical effect')
    return receipt


def plain(value):
    if isinstance(value, list):
        return [plain(x) for x in value]
    if isinstance(value, dict):
        kind = value.get('kind')
        if kind == 'int':
            return int(value['value'])
        if kind == 'real':
            return value['value']
        if kind == 'record':
            return {k: plain(v) for k, v in value['fields'].items()}
        # Preserve enum/unknown provenance; never decode it into numeric zero.
        return {k: plain(v) for k, v in value.items()}
    return value


def evaluate(request, *, review=None, include_receipt=False):
    request = deepcopy(request)
    sources = sources_for(request, review=review)
    raw = ElsaCompiler().compile_project(sources, entry_source_name='main.bangel').to_bytes()
    receipt = execute_jp(raw)
    outputs = plain(receipt['outputs'])
    candidate = outputs['retrieval']
    baseline = candidate['available']
    matched = outputs['match_available']
    kernel = outputs['kernel_review']
    if not kernel['evaluated']:
        kernel['prior_loss'] = kernel['candidate_loss'] = None
    result = dict(profile=PROFILE, name=NAME,
        implementation=dict(id='mj-neural-net', version=VERSION, mode='CLONED_BASELINE',
            baseline_name=BASELINE_NAME, baseline_profile=PROFILE,
            baseline_source_sha256=BASELINE_SOURCE_SHA256),
        status='RELATION_MATCH' if matched else ('RELATION_HOLD' if baseline else 'ASSOCIATIVE_HOLD'),
        r0=dict(marker=outputs['marker'], status=outputs['marker_status'], profile=outputs['marker_profile'],
                biological_state=False, mapped_to_R0=False),
        candidate=candidate['candidate'] if baseline else None,
        baseline_available=baseline, match_available=matched, known_features=outputs['known_features'],
        diagnostic_score=candidate['score'], diagnostic_margin=candidate['margin'],
        weights=outputs['weights'], layers=outputs['layers'], original_cue=outputs['original_cue'],
        placements=outputs['placements'], kernel_review=kernel,
        review_status=outputs['review_status'], advisory_ready=outputs['review_status'] == 'MATCH_FOR_REVIEW',
        physical_actuation_allowed=False, live_trading_allowed=False,
        request_sha256=digest(request), pathway_sha256=outputs['pathway_sha256'],
        jp_sha256=hashlib.sha256(raw).hexdigest(), receipt_root=receipt['receipt_root'],
        source_sha256={name: hashlib.sha256(text.encode()).hexdigest() for name, text in sources.items()},
        resources=receipt['resource_limits'])
    if include_receipt:
        result['receipt'] = receipt
    return result


def example_request():
    return json.loads((HERE / 'examples/partial-cue.json').read_text())


def read_json(path):
    from .transport import load_json_bytes, MAX_BODY_BYTES
    with Path(path).open('rb') as stream:
        return load_json_bytes(stream.read(MAX_BODY_BYTES + 1))


def main(argv=None):
    parser = argparse.ArgumentParser(description=NAME + ' (provisional, simulation profile)')
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('run', 'lower', 'overlay', 'compose', 'lower-overlay'):
        sub = commands.add_parser(name)
        sub.add_argument('request', type=Path)
        sub.add_argument('--review', type=Path)
        sub.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        request = read_json(args.request)
        review = read_json(args.review) if args.review else None
        if args.output.exists():
            raise ValueError('Output already exists')
        if args.command in ('lower', 'lower-overlay'):
            if args.command == 'lower-overlay':
                from .integration import sources_for_overlay
                sources = sources_for_overlay(request, review=review)
            else:
                sources = sources_for(request, review=review)
            raw = ElsaCompiler().compile_project(sources, entry_source_name='main.bangel').to_bytes()
            JPIR.from_bytes(raw).verify()
            args.output.mkdir(parents=True, exist_ok=False)
            for name, text in sources.items():
                (args.output / name).write_text(text)
            (args.output / 'program.jp.json').write_bytes(raw)
            print('Lowered and independently admitted JP')
            return 0
        if args.command in ('overlay', 'compose'):
            from .integration import overlay, compose
            result = (overlay if args.command == 'overlay' else compose)(request, review=review)
        else:
            result = evaluate(request, review=review)
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(json.dumps(result, indent=2) + '\n')
        print(result.get('status', result['profile']) + ' / ' + result['review_status'])
        return 0 if result['advisory_ready'] else 3
    except (ValueError, OSError, RecursionError, BangelDiagnostic) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
