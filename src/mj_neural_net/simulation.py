"""Frozen synthetic comparison. Labels remain outside the model request."""
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from decimal import Decimal
import argparse
import hashlib
import json
from pathlib import Path
import random

from .adapter import HERE, PROFILE, digest, evaluate, example_request

SEED = 230903


def cases():
    rng = random.Random(SEED)
    base = example_request()
    other = deepcopy(base['memories'][0])
    other.update(id='memory-b', entity='robot-b', context='bay-b',
                 pattern=['-0.9', '-0.9', '0.9', '0.9', '-0.9', '0.9', '0.9', '0.9', '-0.9'])
    base['memories'].append(other)
    groups = [('clean', 12), ('partial', 12), ('noisy', 12), ('partial_noisy', 12),
              ('relational_near_miss', 24), ('novel_pattern', 12), ('missing_relation', 12)]
    records = []
    for group, count in groups:
        for _ in range(count):
            index = len(records)
            request = deepcopy(base)
            target = base['memories'][index % 2]
            cue = list(target['pattern'])
            query = request['query']
            query.update(id=f'test-{index:03}', entity=target['entity'], context=target['context'])
            expected = target['id']
            locations = list(range(9))
            rng.shuffle(locations)
            if group in ('noisy', 'partial_noisy'):
                for position in locations[:1 if group == 'partial_noisy' else 2]:
                    cue[position] = '0.9' if cue[position] == '-0.9' else '-0.9'
            if group in ('partial', 'partial_noisy'):
                for position in locations[-3:]:
                    cue[position] = None
            if group == 'relational_near_miss':
                query['context'] = 'unfamiliar-bay'
                expected = None
            if group == 'novel_pattern':
                cue = [rng.choice(('0.9', '-0.9')) for _ in range(9)]
                while cue in [m['pattern'] for m in base['memories']]:
                    cue = [rng.choice(('0.9', '-0.9')) for _ in range(9)]
                query['entity'] = 'unfamiliar-robot'
                expected = None
            if group == 'missing_relation':
                # Ground truth is still positive. Holding costs recall here.
                query['context'] = None
            query['cue'] = cue
            records.append(dict(group=group, expected=expected, request=request))
    return records


def measure(case):
    result = evaluate(case['request'])
    return dict(id=case['request']['query']['id'], group=case['group'], expected=case['expected'],
        baseline=result['candidate'] if result['baseline_available'] else None,
        two_stage=result['candidate'] if result['match_available'] else None,
        score=result['diagnostic_score'], margin=result['diagnostic_margin'],
        status=result['status'], request_sha256=result['request_sha256'],
        jp_sha256=result['jp_sha256'], receipt_root=result['receipt_root'])


def metrics(rows, field, threshold=None):
    tp = fp = tn = fn = held = wrong = 0
    for row in rows:
        predicted = row[field]
        if threshold is not None and Decimal(row['score']) < threshold:
            predicted = None
        expected = row['expected']
        held += predicted is None
        if predicted is not None and predicted == expected:
            tp += 1
        elif predicted is not None:
            fp += 1
            if expected is not None:
                wrong += 1
                fn += 1
        elif expected is None:
            tn += 1
        else:
            fn += 1
    return dict(cases=len(rows), true_matches=tp, false_matches=fp, true_rejections=tn,
        missed_known_matches=fn, wrong_entity_matches=wrong, holds=held,
        precision=tp/(tp+fp) if tp+fp else None,
        recall=tp/(tp+fn) if tp+fn else None,
        coverage=(len(rows)-held)/len(rows))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, choices=(1, 2, 3, 4), default=2)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('Output already exists')
    dataset = cases()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(measure, dataset):
            rows.append(row)
            if len(rows) % 12 == 0:
                print(f'Completed {len(rows)}/{len(dataset)} paired cases', flush=True)
    report = dict(profile='MJ-Neural-Comparison/0.1.0', model_profile=PROFILE, seed=SEED,
        scope='SYNTHETIC_SIMULATION', training_cases=2, test_cases=len(rows),
        dataset_sha256=digest(dataset),
        baseline=metrics(rows, 'baseline'), two_stage=metrics(rows, 'two_stage'),
        baseline_threshold_sweep={str(t): metrics(rows, 'baseline', t) for t in (Decimal('0.8'), Decimal('0.9'), Decimal('1'))},
        by_group={g: {name: metrics([x for x in rows if x['group'] == g], name)
                      for name in ('baseline', 'two_stage')} for g in dict.fromkeys(x['group'] for x in rows)},
        controls=['Same trained weights, cues, placement schedule and associative threshold in both arms',
                  'Only the two-stage arm uses relational metadata; this is an information/filtering ablation',
                  'Ground-truth labels never enter source binding, training, update selection or inference',
                  'Missing relational evidence counts against recall; thresholds fixed before evaluation',
                  'No biological validity, calibrated confidence or generalization is inferred'],
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (HERE/'adapter.py', HERE/'source/neural.bangel', HERE/'PATHWAYS.json', HERE/'simulation.py')},
        cases=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('test_cases', 'baseline', 'two_stage')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
