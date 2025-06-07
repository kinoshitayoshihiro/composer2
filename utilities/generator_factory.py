# utilities/generator_factory.py
from generator.piano_generator import PianoGenerator
from generator.guitar_generator import GuitarGenerator
from generator.bass_generator import BassGenerator
from generator.drum_generator import DrumGenerator
from generator.strings_generator import StringsGenerator
from generator.melody_generator import MelodyGenerator
from generator.sax_generator import SaxGenerator

ROLE_MAP = {
    "melody": MelodyGenerator,
    "counter": MelodyGenerator,
    "pad": StringsGenerator,
    "riff": MelodyGenerator,
    "rhythm": GuitarGenerator,
    "guitar": GuitarGenerator,  # ← これを追加
    "bass": BassGenerator,
    "unison": StringsGenerator,
    "drums": DrumGenerator,
    "piano": PianoGenerator,
}


class GenFactory:
    @staticmethod
    def build_from_config(main_cfg):
        """main_cfg['part_defaults'] を読み取り各 Generator を初期化"""
        global_settings = main_cfg.get("global_settings", {})
        gens = {}
        for part_name, part_cfg in main_cfg["part_defaults"].items():
            role = part_cfg.get("role", part_name)  # role が無ければ楽器名と同じ
            GenCls = ROLE_MAP[role]
            cleaned_part_cfg = dict(part_cfg)
            cleaned_part_cfg.pop("main_cfg", None)
            gens[part_name] = GenCls(
                global_settings=global_settings,
                default_instrument=cleaned_part_cfg.get(
                    "default_instrument", part_name
                ),
                global_tempo=global_settings.get("tempo_bpm"),
                global_time_signature=global_settings.get("time_signature", "4/4"),
                global_key_signature_tonic=global_settings.get("key_tonic"),
                global_key_signature_mode=global_settings.get("key_mode"),
                main_cfg=main_cfg,
                **cleaned_part_cfg,
            )
        return gens
