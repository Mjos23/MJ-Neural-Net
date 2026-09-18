"""Closed composition transport. Projection, diffusion and decisions run in Bangel."""
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from decimal import Decimal

from bangel.compiler import ElsaCompiler

from . import adapter

PROFILE = 'MJ-Neural-Net/0.1.0'
OVERLAY_PROFILE = 'MJ-Neural-Overlay/0.1.0'
REVIEW_PROFILE = 'MJ-Neural-Net-Review/0.1.0'
CATALOG_SHA256 = 'd6d913266a43e644cee7ea810eba0ff2aa5597d5534eb4457fee0c1baf49bf7d'
MAPPINGS_SHA256 = 'd2e1a0c592af83df794477ee8f5a07b7ab440dec574979c64f09f734ba52bdd4'


def example_envelope():
    return {'profile': PROFILE, 'neural': adapter.example_request(), 'memory_recall': None}


def validate(request):
    adapter.shape(request, ('profile', 'neural', 'memory_recall'), 'envelope')
    if request['profile'] != PROFILE:
        raise ValueError('Unsupported MJ Neural Net envelope')
    adapter.validate(request['neural'])
    memory = request['memory_recall']
    if memory is not None:
        from mj_memory_recall.engine import validate as validate_memory
        validate_memory(memory)
        neural = request['neural']
        if neural['as_of'] != memory['as_of']:
            raise ValueError('Composed observations must have the same first-observation time')
        for key in neural['query']:
            if neural['query'][key] != memory['query'][key]:
                raise ValueError('Recall query differs from the original neural query: ' + key)
        original = {item['id']: item for item in neural['memories']}
        recalled = {item['id']: item for item in memory['memories']}
        if set(original) != set(recalled):
            raise ValueError('Recall memory IDs must equal the original neural memory IDs')
        for key, item in original.items():
            if any(recalled[key][field] != value for field, value in item.items()):
                raise ValueError('Recall memory changes an original memory value: ' + key)
    return request


def review_binding(request, review):
    if review is None:
        return False, 'HOLD'
    adapter.shape(review, ('profile', 'request_sha256', 'verdict'), 'envelope review')
    if (review['profile'] != REVIEW_PROFILE or review['request_sha256'] != adapter.digest(request)
            or type(review['verdict']) is not str or review['verdict'] not in ('PASS', 'HOLD', 'DENY')):
        raise ValueError('Independent review must bind the complete MJ Neural Net envelope')
    return True, review['verdict']


def _rebind(request, review, child, profile):
    reviewed, verdict = review_binding(request, review)
    if not reviewed:
        return None
    return {'profile': profile, 'request_sha256': adapter.digest(child), 'verdict': verdict}


@lru_cache(maxsize=1)
def _catalogs():
    items = []
    for name, expected in [('teal-diagrams.json', CATALOG_SHA256), ('teal-mappings.json', MAPPINGS_SHA256)]:
        raw = (adapter.HERE / 'data' / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError('Packaged diagram/source catalog hash mismatch')
        items.append(json.loads(raw))
    catalog, mappings = items
    templates = catalog['templates']
    if len(templates) != 5 or len(catalog['occurrences']) != 100 or len(mappings['entries']) != 500:
        raise ValueError('Incomplete Teal diagram/source coverage')
    if [item['mode'] for item in templates] != list(range(5)):
        raise ValueError('Unknown diagram layer order')
    for template in templates:
        points = template['nodes']
        ids = {p['id'] for p in points}
        if not 1 <= len(points) <= 9 or len(ids) != len(points) or len(template['edges']) > 8:
            raise ValueError('Invalid bounded source geometry')
        if template['approved_sequence'] or template['ordered_executable_path'] is not None:
            raise ValueError('Source reference cannot become an approved traversal')
        for edge in template['edges']:
            if edge['directed'] or edge['from'] not in ids or edge['to'] not in ids:
                raise ValueError('Diagram connector scope or direction changed')
    return catalog, mappings


def catalog():
    diagrams, mappings = _catalogs()
    return deepcopy({'profile': 'MJ-Neural-Net-Catalog/0.1.0', 'diagrams': diagrams,
        'teal_mappings': mappings, 'diagram_sha256': CATALOG_SHA256, 'mappings_sha256': MAPPINGS_SHA256,
        'approved_traversal_claim': False})


def _memory_result(request, review):
    child = request['memory_recall']
    if child is None:
        return None
    from mj_memory_recall import recall
    return recall(child, review=_rebind(request, review, child, 'MJ-Memory-Review/0.1.0'))


def _literal_real(value):
    # Typed serialization of trusted catalog coordinates or runtime outputs.
    text = format(Decimal(str(value)), 'f')
    if not Decimal(text).is_finite():
        raise ValueError('Nonfinite derived scalar')
    return text if '.' in text else text + '.0'


def _point(point):
    return 'overlay.Point(x = ' + _literal_real(point[0]) + ', y = ' + _literal_real(point[1]) + ')'


def _playbook(request, memory_result):
    if not memory_result or not memory_result.get('match_available'):
        return None
    original = next((item for item in request['memory_recall']['memories']
                     if item['id'] == memory_result['candidate']), None)
    if original is None:
        raise ValueError('Recall candidate is not an original memory')
    if original['selection_id'] is None:
        return None
    from mj_memory_recall.engine import catalog_index, selection
    selected = selection(original['selection_id'])
    frame = deepcopy(catalog_index()['frame'])
    if frame != request['memory_recall']['frame'] or len(selected['paths']) != 11:
        raise ValueError('Playbook selection geometry or original frame mismatch')
    return {'selection_id': selected['id'], 'memory_id': original['id'],
        'frame': frame, 'selection': selected,
        'selection_sha256': adapter.digest({'frame': frame, 'selection': selected})}


def _sources(request, review, memory_result=None, baseline=None):
    reviewed, verdict = review_binding(request, review)
    neural = request['neural']
    diagrams, _ = _catalogs()
    previous = adapter.command(neural['query']['previous_command'])
    lines = ['bangel 1.0', 'program mj_neural_net_overlay', 'effects pure',
        'import mj.neural as neural', 'import mj.neural_overlay as overlay',
        'let history: Vector[neural.Transition] = ' + adapter.transitions(neural['history']),
        'let proposed_history: Vector[neural.Transition] = ' + adapter.transitions(neural['history'] + neural['updates']),
        'let validation: Vector[neural.Transition] = ' + adapter.transitions(neural['validation']),
        'let prior_model = neural.train_commands(history)',
        'let proposed_model = neural.candidate_commands(prior_model, proposed_history, ' + str(bool(neural['updates'])).lower() + ')',
        'let kernel_review = neural.review_update(prior_model, proposed_model, validation, ' + str(bool(neural['updates'])).lower() + ')',
        'let independently_reviewed: Bool = ' + str(reviewed).lower(),
        'let verdict: Text = ' + json.dumps(verdict),
        'let update_admitted = kernel_review.promoted and independently_reviewed and verdict == "PASS"',
        'let selected_model = overlay.select_model(prior_model, proposed_model, update_admitted)',
        'let selected_history = overlay.select_history(history, proposed_history, update_admitted)',
        'let forecast = overlay.forecast(selected_model, ' + str(previous) + ')']
    recalled = memory_result or {}
    forecast = recalled.get('forecast') or {}
    forecast_available = bool(forecast.get('available'))
    forecast_node = forecast.get('node') if forecast_available else 0
    probability = forecast.get('probability', '0.0') if forecast_available else '0.0'
    recall_support = forecast.get('support', 0) if forecast_available else 0
    if type(recall_support) is not int or not 0 <= recall_support <= 99:
        raise ValueError('Recall runtime emitted invalid forecast support')
    if forecast_available and (type(forecast_node) is not int or not 1 <= forecast_node <= 9):
        raise ValueError('Recall runtime emitted an invalid forecast node')
    if not Decimal('0') <= Decimal(str(probability)) <= Decimal('1'):
        raise ValueError('Recall runtime emitted an invalid forecast probability')
    lines += ['let recall_forecast_admitted = ' + str(bool(recalled.get('match_available')) and forecast_available).lower() +
              ' and independently_reviewed and verdict == "PASS"',
        'let initial = overlay.incorporate_recall(forecast, ' + str(forecast_node or 0) + ', ' +
            _literal_real(probability) + ', recall_forecast_admitted)']
    layers = []
    projections = []
    projected_edges = []
    for number, template in enumerate(diagrams['templates']):
        prefix = 'g' + str(number)
        points = template['nodes']
        lines += ['let ' + prefix + '_points: Vector[overlay.Point] = [' + ', '.join(_point(p['center_pt']) for p in points) + ']',
                  'let ' + prefix + '_frame = overlay.bounds(' + prefix + '_points)']
        names = {}
        for index, point in enumerate(points):
            name = prefix + '_n' + str(index)
            names[point['id']] = name
            lines.append('let ' + name + ' = overlay.project(' + prefix + '_frame, ' + _point(point['center_pt']) + ')')
        edges = ['overlay.Edge(a = ' + names[e['from']] + ', b = ' + names[e['to']] + ')' for e in template['edges']]
        lines += ['let ' + prefix + '_edges: Vector[overlay.Edge] = [' + ', '.join(edges) + ']',
            'let ' + prefix + '_layer = overlay.diffuse(initial, ' + prefix + '_edges)']
        layers.append(prefix + '_layer')
        lines.append('let ' + prefix + '_nodes: Vector[Int] = [' + ', '.join(names.values()) + ']')
        projections.append(prefix + '_nodes')
        projected_edges.append(prefix + '_edges')
    lines += ['let layers: Vector[neural.Nodes; 5] = [' + ', '.join(layers) + ']',
        'let teal_combined = overlay.combine(initial, layers)']
    playbook = _playbook(request, memory_result)
    extra_sources = {}
    if playbook is not None:
        from mj_memory_recall import engine as memory_engine
        lines += ['import mj.recall as recall', 'import mj.neural_playbook as playbook',
            'let playbook_frame = ' + memory_engine.frame_source(playbook['frame'])]
        path_names = []
        for index, path in enumerate(playbook['selection']['paths']):
            name = 'playbook_actor_' + str(index)
            lines.append('let ' + name + ' = recall.projected_path(' + memory_engine.curve(path) + ', playbook_frame)')
            path_names.append(name)
        lines += ['let playbook_paths: Vector[Result[Vector[Int; 9]]; 11] = [' + ', '.join(path_names) + ']',
            'let playbook_valid = playbook.all_projected(playbook_paths)',
            'let playbook_admitted = playbook_valid and independently_reviewed and verdict == "PASS"',
            'let playbook_scores = playbook.diffuse(initial, playbook_paths)',
            'let combined = playbook.combine(initial, layers, playbook_scores, playbook_admitted)',
            'emit playbook_paths = playbook_paths', 'emit playbook_valid = playbook_valid',
            'emit playbook_admitted = playbook_admitted', 'emit playbook_scores = playbook_scores',
            'emit playbook_selection_sha256 = ' + json.dumps(playbook['selection_sha256'])]
        extra_sources = {'recall.bangel': (memory_engine.HERE / 'source/recall.bangel').read_text(),
                         'playbook.bangel': (adapter.HERE / 'source/playbook.bangel').read_text()}
    else:
        lines.append('let combined = teal_combined')
    lines += [
        'let command_support = overlay.combined_support(overlay.support(selected_history, ' + str(previous) + '), ' + str(recall_support) + ', recall_forecast_admitted)',
        'let placement = overlay.placement(combined, command_support)',
        'emit forecast = forecast', 'emit initial = initial', 'emit layers = layers',
        'emit combined = combined', 'emit teal_combined = teal_combined', 'emit placement = placement',
        'emit projected_nodes = [' + ', '.join(projections) + ']',
        'emit projected_edges = [' + ', '.join(projected_edges) + ']',
        'emit kernel_review = kernel_review', 'emit kernel_update_admitted = update_admitted',
        'emit recall_forecast_admitted = recall_forecast_admitted',
        'emit review_status = neural.downstream_status(placement.available, independently_reviewed, verdict)',
        'emit request_sha256 = ' + json.dumps(adapter.digest(request)),
        'emit catalog_sha256 = ' + json.dumps(CATALOG_SHA256)]
    if memory_result is not None:
        lines += ['emit recall_request_sha256 = ' + json.dumps(memory_result['request_sha256']),
                  'emit recall_receipt_root = ' + json.dumps(memory_result['receipt_root'])]
    if baseline is not None:
        lines += ['emit baseline_receipt_root = ' + json.dumps(baseline['receipt_root'])]
        lines += ['let decision = overlay.decide(' + ', '.join([
            json.dumps(baseline['candidate'] or ''), str(baseline['match_available']).lower(),
            json.dumps(recalled.get('candidate') or ''), str(bool(recalled.get('match_available'))).lower(),
            'independently_reviewed', 'verdict',
            str(baseline['review_status'] == 'DOWNSTREAM_DENY').lower(),
            str(recalled.get('review_status') == 'DOWNSTREAM_DENY').lower()]) + ')',
            'emit decision = decision']
    # Imports must precede executable declarations in the generated module.
    ordered = [line for line in lines if line.startswith(('bangel ', 'program ', 'effects ', 'import '))]
    ordered += [line for line in lines if line.startswith('let ')]
    ordered += [line for line in lines if line.startswith('emit ')]
    return {'neural.bangel': (adapter.HERE / 'source/neural.bangel').read_text(),
            'overlay.bangel': (adapter.HERE / 'source/overlay.bangel').read_text(),
            'main.bangel': '\n'.join(ordered) + '\n', **extra_sources}


def sources_for_overlay(request, *, review=None):
    request = deepcopy(request)
    validate(request)
    review_binding(request, review)
    return _sources(request, review, _memory_result(request, review))


def _execute(request, review, memory_result=None, baseline=None, include_receipt=False):
    sources = _sources(request, review, memory_result, baseline)
    raw = ElsaCompiler().compile_project(sources, entry_source_name='main.bangel').to_bytes()
    receipt = adapter.execute_jp(raw)
    outputs = adapter.plain(receipt['outputs'])
    placement = outputs['placement']
    if not placement['available']:
        placement['node'] = None
    result = {'profile': OVERLAY_PROFILE, 'request_sha256': adapter.digest(request),
        'catalog_sha256': CATALOG_SHA256, 'mappings_sha256': MAPPINGS_SHA256,
        'source_template_ids': [t['id'] for t in _catalogs()[0]['templates']],
        'forecast': outputs['forecast'], 'initial': outputs['initial'],
        'layers': outputs['layers'], 'teal_combined': outputs['teal_combined'],
        'combined': outputs['combined'], 'placement': placement, 'playbook_layer': None,
        'projected_nodes': outputs['projected_nodes'], 'projected_edges': outputs['projected_edges'],
        'kernel_review': outputs['kernel_review'], 'kernel_update_admitted': outputs['kernel_update_admitted'],
        'recall_forecast_admitted': outputs['recall_forecast_admitted'],
        'review_status': outputs['review_status'], 'advisory_ready': outputs['review_status'] == 'MATCH_FOR_REVIEW',
        'projection_profile': 'MJ-Five-Template-Diffusion/0.1.0',
        'source_directions_claimed': False, 'approved_traversal_claim': False,
        'physical_actuation_allowed': False, 'live_trading_allowed': False,
        'jp_sha256': hashlib.sha256(raw).hexdigest(), 'receipt_root': receipt['receipt_root'],
        'resources': receipt['resource_limits'],
        'source_sha256': {k: hashlib.sha256(v.encode()).hexdigest() for k, v in sources.items()}}
    if not result['kernel_review']['evaluated']:
        result['kernel_review']['prior_loss'] = result['kernel_review']['candidate_loss'] = None
    if memory_result is not None:
        result['recall_receipt_root'] = memory_result['receipt_root']
    playbook = _playbook(request, memory_result)
    if playbook is not None:
        projected = outputs['playbook_paths']
        if not outputs['playbook_valid'] or any(item.get('variant') != 'Ok' for item in projected):
            raise ValueError('Native projection rejected original playbook geometry')
        selected = playbook['selection']
        result['playbook_layer'] = {
            'profile': 'MJ-Neural-Playbook-Layer/0.1.0', 'available': True,
            'admitted': outputs['playbook_admitted'], 'memory_id': playbook['memory_id'],
            'selection_id': playbook['selection_id'], 'selection_sha256': playbook['selection_sha256'],
            'frame': playbook['frame'], 'source': selected['source'],
            'paths': [{**deepcopy(path), 'nodes': node['value']} for path, node in zip(selected['paths'], projected)],
            'sampling': 'quadratic Bezier t=step/8; step 0..8; repeated nodes preserved',
            'edge_rule': 'adjacent samples in both orientations; new engineered diffusion',
            'scores': outputs['playbook_scores'], 'node_namespace': 'MJ.Engineered.Spatial.N1-N9',
            'source_directions_claimed': False, 'approved_traversal_claim': False}
    if include_receipt:
        result['receipt'] = receipt
    return result, outputs.get('decision')


def overlay(request, *, review=None, include_receipt=False):
    request = deepcopy(request)
    validate(request)
    review_binding(request, review)
    return _execute(request, review, _memory_result(request, review), include_receipt=include_receipt)[0]


def compose(request, *, review=None, include_receipt=False):
    request = deepcopy(request)
    validate(request)
    review_binding(request, review)
    baseline = adapter.evaluate(request['neural'], review=_rebind(request, review, request['neural'],
        'MJ-Neural-Review/0.1.0'), include_receipt=include_receipt)
    recalled = _memory_result(request, review)
    layers, decision = _execute(request, review, recalled, baseline, include_receipt=include_receipt)
    return {'profile': PROFILE, 'name': adapter.NAME, 'request_sha256': adapter.digest(request),
        'candidate': decision['candidate'] if decision['matched'] else None,
        'match_available': decision['matched'], 'candidate_conflict': decision['conflict'],
        'review_status': decision['review_status'], 'advisory_ready': decision['review_status'] == 'MATCH_FOR_REVIEW',
        'baseline': baseline, 'overlay': layers, 'memory_recall': recalled,
        'r0': baseline['r0'], 'physical_actuation_allowed': False, 'live_trading_allowed': False,
        'receipt_root': layers['receipt_root']}
