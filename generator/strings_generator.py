# generator/strings_generator.py
from .melody_generator import MelodyGenerator
from music21 import volume


class StringsGenerator(MelodyGenerator):
    def __init__(self, *args, **kwargs):
        # Pop any instrument_name passed from config to avoid duplicate
        config_instrument = kwargs.pop("instrument_name", None)
        # Enforce default instrument for this generator
        default_name = "String Ensemble"
        instrument_name = config_instrument or default_name
        super().__init__(*args, instrument_name=instrument_name, **kwargs)

    def compose(self, section_data, next_section_data=None):
        # 既存の compose ロジックを継承
        return super().compose(section_data, next_section_data)

    def _postprocess_stream(self, part):
        """pad 用: 音長を 4 倍・legato・低 velocity など簡易調整"""
        for n in part.flat.notes:
            n.quarterLength *= 4
            if n.volume is None:
                n.volume = volume.Volume()
            n.volume.velocity = max(30, (n.volume.velocity or 64) - 20)
        profile_name = (
            self.cfg.get("humanize_profile")
            or section_data.get("humanize_profile")
            or self.global_settings.get("humanize_profile")
        )
        # パートごと
        if profile_name:
            humanizer.apply(part, profile_name)

        # スコア全体
        if global_profile:
            humanizer.apply(score, global_profile)
        return part
