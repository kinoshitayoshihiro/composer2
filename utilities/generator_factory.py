# utilities/generator_factory.py
from generator.piano_generator import PianoGenerator
from generator.guitar_generator import GuitarGenerator
from generator.bass_generator import BassGenerator
from generator.drum_generator import DrumGenerator
from generator.strings_generator import StringsGenerator
from generator.melody_generator import MelodyGenerator
from generator.sax_generator import SaxGenerator
from generator.base_part_generator import BasePartGenerator
from music21 import instrument as m21instrument

class GenFactory:
    @staticmethod
    def build_from_config(main_cfg, rhythm_lib=None):
        """main_cfg['part_defaults'] を読み取り各 Generator を初期化

        Parameters
        ----------
        main_cfg : dict
            Parsed configuration dictionary from ``load_main_cfg``.
        rhythm_lib : RhythmLibrary | None
            Optional rhythm library object providing pattern dictionaries for
            each part. If provided, the corresponding pattern set is passed to
            each generator via ``part_parameters``.
        """
        global_settings = main_cfg.get("global_settings", {})
        gens = {}
        for part_name, part_cfg in main_cfg["part_defaults"].items():
            role = part_cfg.get("role", part_name)  # role が無ければ楽器名と同じ
            GenCls = ROLE_DISPATCH[role]
            cleaned_part_cfg = dict(part_cfg)
            cleaned_part_cfg.pop("main_cfg", None)
            inst_spec = cleaned_part_cfg.get("default_instrument", part_name)
            if isinstance(inst_spec, str):
                try:
                    inst_obj = m21instrument.fromString(inst_spec)
                except Exception:
                    try:
                        inst_obj = m21instrument.fromString(part_name)
                    except Exception:
                        inst_obj = m21instrument.Percussion()
            else:
                inst_obj = inst_spec

            lib_params = {}
            if rhythm_lib is not None:
                if part_name == "drums":
                    lib_params = getattr(rhythm_lib, "drum_patterns", {}) or {}
                elif part_name == "bass":
                    lib_params = getattr(rhythm_lib, "bass_patterns", {}) or {}
                elif part_name == "piano":
                    lib_params = getattr(rhythm_lib, "piano_patterns", {}) or {}
                elif part_name in ("guitar", "rhythm"):
                    lib_params = getattr(rhythm_lib, "guitar", {}) or {}

            if lib_params and not isinstance(next(iter(lib_params.values()), {}), dict):
                lib_params = {
                    k: v.model_dump() if hasattr(v, "model_dump") else dict(v)
                    for k, v in lib_params.items()
                }

            part_params = cleaned_part_cfg.get("part_parameters", {})
            if lib_params:
                part_params = {**lib_params, **part_params}
            cleaned_part_cfg["part_parameters"] = part_params

            gens[part_name] = GenCls(
                global_settings=global_settings,
                default_instrument=inst_obj,
                part_name=part_name,
                global_tempo=global_settings.get("tempo_bpm"),
                global_time_signature=global_settings.get("time_signature", "4/4"),
                global_key_signature_tonic=global_settings.get("key_tonic"),
                global_key_signature_mode=global_settings.get("key_mode"),
                main_cfg=main_cfg,
                **cleaned_part_cfg,
            )
        return gens


# === Explicit role → generator mapping ===
ROLE_DISPATCH: dict[str, type[BasePartGenerator]] = {
    "melody": MelodyGenerator,
    "counter": MelodyGenerator,
    "pad": StringsGenerator,
    "riff": MelodyGenerator,
    "rhythm": GuitarGenerator,
    "guitar": GuitarGenerator,
    "bass": BassGenerator,
    "unison": StringsGenerator,
    "drums": DrumGenerator,
    "piano": PianoGenerator,
    "strings": StringsGenerator,
    "sax": SaxGenerator,
}
