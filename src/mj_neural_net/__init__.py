"""MJ Neural Net: an independently installable clone of the Bangel neural baseline."""
from .adapter import NAME, PROFILE, VERSION, evaluate, example_request, sources_for
from .api import create_app
from .integration import compose, overlay, catalog, example_envelope, sources_for_overlay

__version__ = VERSION
__all__ = ['NAME', 'PROFILE', 'VERSION', 'evaluate', 'example_request', 'sources_for', 'create_app', 'compose', 'overlay', 'catalog', 'example_envelope', 'sources_for_overlay']
