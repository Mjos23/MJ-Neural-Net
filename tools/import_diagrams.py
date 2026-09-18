"""Import an audited source snapshot with explicit caller-supplied name migration.

No graph, coordinate, or edge direction is inferred by this helper. Historical
source hashes remain unchanged; normalized display metadata gets its own hash.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re


def normalize(value, replacements):
    if isinstance(value, str):
        for old, new in replacements.items():
            if not old or not isinstance(new, str):
                raise ValueError('Replacement keys and values must be nonempty text')
            def substitute(match):
                word = match.group()
                return new.upper() if word.isupper() else (new.lower() if word.islower() else new)
            value = re.sub(re.escape(old), substitute, value, flags=re.IGNORECASE)
        return value
    if isinstance(value, list):
        return [normalize(item, replacements) for item in value]
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            target = normalize(key, replacements)
            if target in result:
                raise ValueError('Name migration would collide dictionary keys')
            result[target] = normalize(item, replacements)
        return result
    return value


def import_snapshot(source, target, replacements):
    raw = source.read_bytes()
    snapshot = normalize(json.loads(raw), replacements)
    snapshot['display_normalization'] = {
        'method': 'EXPLICIT_CALLER_SUPPLIED_HISTORICAL_PRODUCT_NAME_MIGRATION',
        'source_snapshot_sha256': hashlib.sha256(raw).hexdigest(),
        'canonical_names': sorted(set(replacements.values())),
        'geometry_modified': False, 'edge_directions_added': False,
        'source_approval_created': False,
    }
    target.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('target', type=Path)
    parser.add_argument('--replacement-file', required=True, type=Path)
    args = parser.parse_args()
    replacements = json.loads(args.replacement_file.read_text())
    if type(replacements) is not dict:
        raise ValueError('Replacement file must be a JSON object')
    import_snapshot(args.source, args.target, replacements)


if __name__ == '__main__':
    main()
