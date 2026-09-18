"""Minimal WSGI API. Review is supplied by a host callback outside client JSON."""
from copy import deepcopy
import json

from bangel.diagnostics import BangelDiagnostic

from .adapter import (
    BASELINE_NAME, BASELINE_SOURCE_SHA256, NAME, PROFILE, VERSION,
    evaluate, review_binding, validate,
)
from .transport import MAX_BODY_BYTES, load_json_bytes
from . import integration

HTTP_PROFILE = 'MJ-Neural-Net-HTTP/0.1.0'
MAX_RESPONSE_BYTES = 1048576


def profile():
    return {
        'name': NAME, 'version': VERSION, 'http_profile': HTTP_PROFILE,
        'request_profile': PROFILE, 'status': 'PROVISIONAL',
        'baseline': {'name': BASELINE_NAME, 'source_sha256': BASELINE_SOURCE_SHA256,
                     'mode': 'CLONED_BASELINE'},
        'routes': ['GET /v1/profile', 'GET /v1/catalog', 'POST /v1/evaluate', 'POST /v1/overlay', 'POST /v1/compose'],
        'compose_profile': integration.PROFILE, 'overlay_profile': integration.OVERLAY_PROFILE,
        'limits': {'request_bytes': MAX_BODY_BYTES, 'response_bytes': MAX_RESPONSE_BYTES},
        'review': 'SERVER_CALLBACK_ONLY; absent review remains HOLD',
        'r0': {'marker': 'r0', 'profile': 'MJ-Neural-r0/0.1.0', 'status': 'PROVISIONAL',
               'biological_state': False, 'mapped_to_R0': False},
        'pathway_status': 'FIVE_SOURCE_REFERENCE_TEMPLATES_NEW_ENGINEERED_PROJECTION',
        'diagram_coverage': {'teal_papers': 500, 'yellow_instances': 100, 'templates': 5},
        'optional_playbook_layer': {'profile': 'MJ-Neural-Playbook-Layer/0.1.0',
            'actors': 11, 'samples_per_actor': 9, 'admission': 'MATCHED_MEMORY_AND_WHOLE_ENVELOPE_PASS'},
        'review_profiles': {'evaluate': 'MJ-Neural-Review/0.1.0', 'compose_and_overlay': integration.REVIEW_PROFILE},
        'physical_actuation_allowed': False, 'live_trading_allowed': False,
    }


def _response(start_response, status, value, extra=()):
    encoded = json.dumps(value, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')
    if len(encoded) > MAX_RESPONSE_BYTES:
        status = '500 Internal Server Error'
        encoded = b'{"error":{"code":"OUTPUT_LIMIT","message":"Result exceeds the API response bound"}}'
    headers = [('Content-Type', 'application/json; charset=utf-8'),
               ('Content-Length', str(len(encoded))), ('Cache-Control', 'no-store'),
               ('X-Content-Type-Options', 'nosniff')]
    start_response(status, headers + list(extra))
    return [encoded]


def _error(start_response, status, code, message, extra=()):
    return _response(start_response, status, {'error': {'code': code, 'message': message}}, extra)


def create_app(*, review_provider=None):
    """Return a WSGI application without starting a server.

    ``review_provider`` is a trusted host callable receiving a deep copy of a
    validated request. It returns None or the existing hash-bound review record.
    For compose/overlay, review binds the complete envelope before any child review
    is rebound. The callback is not configurable through HTTP, and client input cannot carry
    review fields. The host owns reviewer authentication and network exposure.
    """
    if review_provider is not None and not callable(review_provider):
        raise TypeError('review_provider must be callable or None')

    def application(environ, start_response):
        path = environ.get('PATH_INFO', '')
        method = environ.get('REQUEST_METHOD', '')
        if path not in ('/v1/profile', '/v1/catalog', '/v1/evaluate', '/v1/overlay', '/v1/compose'):
            return _error(start_response, '404 Not Found', 'NOT_FOUND', 'Unknown API route')
        allowed = 'GET' if path in ('/v1/profile', '/v1/catalog') else 'POST'
        if method != allowed:
            return _error(start_response, '405 Method Not Allowed', 'METHOD_NOT_ALLOWED',
                          'Method not allowed for this route', [('Allow', allowed)])
        if environ.get('QUERY_STRING', ''):
            return _error(start_response, '400 Bad Request', 'QUERY_NOT_ALLOWED',
                          'Query parameters are not part of this closed profile')
        if path == '/v1/profile':
            return _response(start_response, '200 OK', profile())

        if path == '/v1/catalog':
            try:
                return _response(start_response, '200 OK', integration.catalog())
            except (ValueError, OSError):
                return _error(start_response, '503 Service Unavailable', 'CATALOG_UNAVAILABLE', 'Packaged source catalog failed integrity verification')
        validator = validate if path == '/v1/evaluate' else integration.validate
        binder = review_binding if path == '/v1/evaluate' else integration.review_binding
        runner = {'/v1/evaluate': evaluate, '/v1/overlay': integration.overlay, '/v1/compose': integration.compose}[path]

        media = [part.strip().lower() for part in environ.get('CONTENT_TYPE', '').split(';')]
        if not media or media[0] != 'application/json' or any(
                part not in ('charset=utf-8', 'charset="utf-8"') for part in media[1:]):
            return _error(start_response, '415 Unsupported Media Type', 'UNSUPPORTED_MEDIA_TYPE',
                          'Use application/json with UTF-8 encoding')
        length_text = environ.get('CONTENT_LENGTH', '')
        if not length_text:
            return _error(start_response, '411 Length Required', 'LENGTH_REQUIRED',
                          'A bounded Content-Length is required')
        if type(length_text) is not str or not length_text.isascii() or not length_text.isdigit():
            return _error(start_response, '400 Bad Request', 'INVALID_LENGTH', 'Invalid Content-Length')
        if len(length_text) > 5 or int(length_text) > MAX_BODY_BYTES:
            return _error(start_response, '413 Content Too Large', 'INPUT_LIMIT',
                          'Input exceeds 65536 bytes')
        length = int(length_text)
        try:
            raw = environ['wsgi.input'].read(length)
            if type(raw) is not bytes or len(raw) != length:
                raise ValueError('Body length does not match Content-Length')
            request = load_json_bytes(raw)
            validator(request)
        except (KeyError, OSError, UnicodeError, ValueError, RecursionError):
            return _error(start_response, '400 Bad Request', 'INVALID_REQUEST',
                          'Body does not satisfy the closed neural request profile')

        try:
            review = None if review_provider is None else review_provider(deepcopy(request))
            binder(request, review)
        except Exception:
            return _error(start_response, '503 Service Unavailable', 'REVIEW_UNAVAILABLE',
                          'Independent review is unavailable or invalid; no advisory is issued')
        try:
            result = runner(request, review=review, include_receipt=False)
        except (ValueError, RecursionError, BangelDiagnostic):
            return _error(start_response, '422 Unprocessable Content', 'EXECUTION_REJECTED',
                          'Neural execution did not satisfy the existing runtime contract')
        except Exception:
            return _error(start_response, '500 Internal Server Error', 'EXECUTION_FAILED',
                          'Neural execution failed; no advisory is issued')
        return _response(start_response, '200 OK', result)

    return application


application = create_app()
