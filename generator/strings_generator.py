# generator/strings_generator.py
from music21 import volume

from .melody_generator import MelodyGenerator


class StringsGenerator(MelodyGenerator):
    def __init__(self, *args, main_cfg=None, **kwargs):
        self.main_cfg = main_cfg
        # Pop any instrument_name passed from config to avoid duplicate
        config_instrument = kwargs.pop("instrument_name", None)
        # Enforce default instrument for this generator
        default_name = "String Ensemble"
        instrument_name = config_instrument or default_name
        super().__init__(*args, instrument_name=instrument_name, **kwargs)

    def compose(
        self,
        *,
        section_data,
        overrides_root=None,
        groove_profile_path=None,
        next_section_data=None,
        part_specific_humanize_params=None,
        shared_tracks=None,
    ):
        """Delegate to MelodyGenerator.compose while ignoring extra args."""
        _ = overrides_root, groove_profile_path, next_section_data, part_specific_humanize_params, shared_tracks
        return super().compose(section_data)

    def _postprocess_stream(self, part):
        """pad 用: 音長を 4 倍・legato・低 velocity など簡易調整"""
        for n in part.flatten().notes:
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
            humanizer.apply(
                part,
                profile_name,
                global_settings=self.global_settings,
            )

        # スコア全体
        if global_profile:
            humanizer.apply(
                score,
                global_profile,
                global_settings=self.global_settings,
            )
        return part
