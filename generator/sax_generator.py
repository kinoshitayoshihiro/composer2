from .melody_generator import MelodyGenerator
from music21 import instrument


class SaxGenerator(MelodyGenerator):
    """Melody generator preset for alto saxophone."""

    def __init__(self, **kwargs):
        kwargs.setdefault("instrument_name", "Alto Saxophone")
        super().__init__(**kwargs)
