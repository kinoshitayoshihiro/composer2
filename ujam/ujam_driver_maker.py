# ujam/ujam_driver_maker.py
from __future__ import annotations
import argparse, math, csv, statistics as st, os
from typing import List, Dict, Tuple, Optional
import pretty_midi as pm
import yaml

# Genre-oriented presets for Mod (lower = more open / longer ring)
MOD_PRESETS = {
    # very open ballad / cinematic clean
    "open_ballad":   {"A": 12, "B": 18, "Build": 28, "default": 16},
    # pop open but a bit tighter than ballad
    "pop_open":      {"A": 14, "B": 22, "Build": 32, "default": 18},
    # rock lower (opens more than default, still punchy)
    "rock_low":      {"A": 18, "B": 28, "Build": 48, "default": 24},
    # ambient / post-rock very open
    "ambient_open":  {"A":  8, "B": 12, "Build": 20, "default": 10},
    # metal tight (more damp by default / higher values)
    "metal_tight":   {"A": 30, "B": 42, "Build": 64, "default": 36},
    # funk crisp but not choppy
    "funk_open":     {"A": 16, "B": 24, "Build": 36, "default": 20},
}


# ===== optional audio onset (librosa) =====
def audio_onsets_to_dummy_notes(audio_path: str):
    """
    Returns: (dummy_notes, duration_sec, est_tempo, beat_times)
    """
    try:
        import librosa, numpy as np
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        dur = float(librosa.get_duration(y=y, sr=sr))

        # beat 検出（秒）
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
        beat_times = np.asarray(beats, dtype=float) if beats is not None else np.array([])

        # onset 検出
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onsets = librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=sr, units="time",
            backtrack=True, pre_max=8, post_max=8, pre_avg=16, post_avg=16, delta=0.05
        )

        # フォールバック：onset が少なければ beat から 1/2 拍で補完
        if beat_times.size:
            half_beats = []
            for i in range(len(beat_times)-1):
                t0, t1 = beat_times[i], beat_times[i+1]
                half_beats.extend([t0, (t0+t1)/2.0])
            # 結合して一意に
            onsets = np.unique(np.concatenate([onsets, np.asarray(half_beats, dtype=float)]))

        notes = [pm.Note(velocity=96, pitch=64, start=float(t), end=float(t)+0.05) for t in onsets]
        est_tempo = float(tempo) if tempo and tempo > 0 else 120.0
        print(f"[INFO] audio_onsets={len(onsets)} beats={len(beat_times)} tempo_est={est_tempo:.1f}")
        return notes, dur, est_tempo, beat_times
    except Exception:
        return [], 0.0, 120.0, None


# ===== phrase / style maps =====
def build_common_phrase_map(c0: int, names_in_ui_order: List[str]) -> Dict[str, int]:
    return {name: c0 + i for i, name in enumerate(names_in_ui_order)}


def build_style_phrase_map(c1: int) -> Dict[str, int]:
    return {f"style_{i+1}": c1 + i for i in range(12)}  # C1..B1


# ===== utility =====
def bars_iter(total_dur: float, bar_len: float):
    n_bars = int(math.ceil(total_dur / bar_len + 1e-6))
    for b in range(n_bars):
        yield b, b * bar_len, (b + 1) * bar_len


def notes_in_window(notes, t0, t1):
    return [n for n in notes if not (n.end <= t0 or n.start >= t1)]


def active_ratio(notes, t0, t1):
    """[t0,t1) に重なる有音時間の合計 / 区間長"""
    if t1 <= t0:
        return 0.0
    tot = 0.0
    for n in notes:
        a = max(t0, n.start)
        b = min(t1, n.end)
        if b > a:
            tot += (b - a)
    return tot / (t1 - t0)


def beat_windows(t0, t1, beats_per_bar=4, subdiv=1):
    span = (t1 - t0) / (beats_per_bar * max(1, subdiv))
    k = beats_per_bar * max(1, subdiv)
    for i in range(k):
        yield t0 + i * span, t0 + (i + 1) * span


def build_segments_from_notes(notes, t0, t1, min_rest=0.08):
    """区間[t0,t1)を、ソースノートのon/off境界で分割し、各セグメントが有音か休符かを返す。"""
    if t1 <= t0:
        return []
    cuts = {t0, t1}
    for n in notes:
        if n.end > t0 and n.start < t1:
            cuts.add(max(t0, n.start))
            cuts.add(min(t1, n.end))
    ts = sorted(cuts)
    segs = []
    for a, b in zip(ts[:-1], ts[1:]):
        if b - a <= 1e-5:
            continue
        active = any((n.start < b and n.end > a) for n in notes)
        if (b - a) < min_rest and not active:
            continue
        segs.append((a, b, active))
    return segs


def sixteenth_grid(bar_len: float) -> float:
    return bar_len / 16.0


def offbeat_ratio_16th(notes, bar_len):
    if not notes:
        return 0.0
    six = sixteenth_grid(bar_len)
    qpos = [round(((n.start % bar_len) / six)) for n in notes]
    return sum(1 for q in qpos if q % 2 == 1) / max(1, len(qpos))


def swing_ratio_16th(notes, bar_len):
    if not notes:
        return 0.0
    six = sixteenth_grid(bar_len)
    offs = []
    for n in notes:
        q = (n.start % bar_len) / six
        frac = q - round(q)
        offs.append(abs(frac))
    if not offs:
        return 0.0
    return min(0.25, st.median(offs)) / 0.25  # 0..1


def energy(notes):
    if not notes:
        return 0.0
    return len(notes) + 0.005 * sum(n.velocity for n in notes)


def detect_sections(bar_energies: List[float], win: int = 4) -> List[str]:
    sm = []
    for i in range(len(bar_energies)):
        s = bar_energies[max(0, i - win + 1) : i + 1]
        sm.append(sum(s) / max(1, len(s)))
    if not sm:
        return []
    qs = st.quantiles(sm, n=4) if len(set(sm)) > 1 else [0, 0, 0]
    lo, hi = qs[1], qs[2] if len(qs) >= 3 else (0.0, 0.0)
    bands = []
    for v in sm:
        if v < lo:
            bands.append("A")
        elif v < hi:
            bands.append("B")
        else:
            bands.append("Build")
    return bands


def build_bars_from_beats(beat_times, beats_per_bar=4, dur=None):
    """beat列（秒）から [ (t0,t1), ... ] の小節境界列を作る（dur まで伸ばす）"""
    import statistics as _st

    bt = list(map(float, list(beat_times or [])))
    bars = []
    for i in range(0, max(0, len(bt) - beats_per_bar), beats_per_bar):
        t0 = bt[i]
        t1 = bt[i + beats_per_bar]
        if t1 > t0:
            bars.append((t0, t1))
    # 末尾を dur まで拡張
    if dur and bars:
        avg = _st.median([b[1] - b[0] for b in bars]) if len(bars) >= 2 else (bars[-1][1] - bars[-1][0])
        end = bars[-1][1]
        while (end + 0.25 * avg) < dur:   # 少し手前まで増やす
            bars.append((end, end + avg))
            end += avg
    return bars


def fold_to_range(p, lo, hi):
    while p < lo:
        p += 12
    while p > hi:
        p -= 12
    return p


def power_chord_from_root(root, lo=60, hi=72):
    r = fold_to_range(root, lo, hi)
    pcs = [r, r + 7, r + 12]
    return [p for p in pcs if p <= hi]


def detect_root_midi_simple(notes_bar):
    # 最低音をルート代理に（music21省略版）
    return min(notes_bar, key=lambda n: (n.start, n.pitch)).pitch if notes_bar else 60


def _emit_chord_block(drv, cfg, root_use, sa, sb, start_off=0.01, end_off=0.02):
    lo, hi = cfg["ujam"]["chord_low"], cfg["ujam"]["chord_high"]
    for p in power_chord_from_root(root_use, lo, hi):
        drv.notes.append(pm.Note(velocity=100, pitch=p, start=sa + start_off, end=sb - end_off))


# ===== phrase choice =====
def choose_phrase(notes_bar, bar_len, rules, section_label):
    if not notes_bar:
        return "silence"
    n = len(notes_bar)
    durs = sorted([x.end - x.start for x in notes_bar])
    med = durs[len(durs) // 2]
    off = offbeat_ratio_16th(notes_bar, bar_len)

    if med > rules["long_med_ratio"] * bar_len and n <= 3:
        return "generic_rhythm_sustain"

    if n >= rules["dense_eighth_notes"]:
        if med < 0.10 * bar_len:
            return "muted_1_8"
        if off > rules["offbeat_threshold"]:
            return "off_beat_1_8"
        return "open_1_8"

    if rules["quarter_range"][0] <= n <= rules["quarter_range"][1]:
        return "open_stops_1_4"

    if section_label == "Build":
        return "build_up_open_1_8"

    return "generic_chord_rhythm"


# ===== swing microtiming =====
def apply_swing_microtiming(note_list: List[pm.Note], bar_len: float, amount01: float):
    """偶数16分（裏）を後ろへズラす。amount01=0..1"""
    if amount01 <= 0:
        return
    six = sixteenth_grid(bar_len)
    delay = amount01 * six * 0.5  # 最大で裏16分の半分だけ後ろへ
    for n in note_list:
        q = round(((n.start % bar_len) / six))
        if q % 2 == 1:  # 裏
            n.start += delay
            n.end += delay


# ===== vocal synchro =====
def load_synchro_csv(path: str) -> List[Tuple[float, str]]:
    evts = []
    with open(path, newline="") as f:
        for t, typ in csv.reader(f):
            try:
                evts.append((float(t), typ.strip()))
            except Exception:
                pass
    return sorted(evts, key=lambda x: x[0])


def nearest_grid_time(t: float, bar_len: float, div: int = 8) -> float:
    step = bar_len / div
    k = round(t / step)
    return k * step


# ===== main =====
def main():
    ap = argparse.ArgumentParser(description="Suno → UJAM driver MIDI (IRON2)")
    ap.add_argument("input", help="MIDI or audio path")
    ap.add_argument("out_mid")
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    ap.add_argument("--synchro", default=None, help="CSV(time_sec,type) for vocal sync")
    ap.add_argument("--force-bpm", type=float, default=None, help="BPMを固定（推定を無視）")
    ap.add_argument("--beats-per-bar", type=int, default=4, help="1小節の拍数（既定4/4）")
    ap.add_argument("--min-bars", type=int, default=8, help="この本数未満ならテンポ等間隔へフォールバック")
    ap.add_argument("--force-seconds", type=float, default=None, help="曲の総尺（秒）を強制")
    ap.add_argument("--tail-pad", type=float, default=1.5, help="終端に追加する余白秒（既定1.5s）")
    ap.add_argument("--debug-bars", action="store_true",
                    help="小節頭にC6の点を打って可視化（既定OFF）")
    ap.add_argument("--gate-by-source", action="store_true",
                    help="コードをソースMIDIのノート長でゲート（休符は鳴らさない）")
    ap.add_argument("--preserve-rests", action="store_true",
                    help="休符区間で Common 'silence' を自動トリガ")
    ap.add_argument("--min-rest", type=float, default=0.08,
                    help="休符として扱う最小長(秒)")
    ap.add_argument("--audio-gate-8th", type=float, default=1.0,
                    help="WAV入力時、オンセット周辺を 1/8音符×係数 だけ有音とみなす(0で無効)")
    ap.add_argument("--segment-root", action="store_true",
                    help="有音セグメントごとにルートを再判定（1小節内の転和音に追従）")
    ap.add_argument("--min-chord-len", type=float, default=0.12,
                    help="コード1回の最小長(秒)。短いものは捨てる/結合")
    ap.add_argument("--chord-retrigger-gap", type=float, default=0.06,
                    help="この隙間以下はリトリガせず延長扱い")
    ap.add_argument("--chord-grid", type=int, default=8,
                    help="コードの開始/終了をこの分割にスナップ(0で無効, 8 or 16推奨)")
    ap.add_argument("--chord-hold", choices=["bar", "beat", "segment"], default="bar",
                    help="コード保持の粒度。bar=小節単位, beat=拍単位, segment=細分(従来のゲート)")
    ap.add_argument("--bar-rest-thresh", type=float, default=0.18,
                    help="小節を“無音”とみなすカバレッジ閾値(0..1)")
    ap.add_argument("--beat-rest-thresh", type=float, default=0.25,
                    help="拍を“無音”とみなすカバレッジ閾値(0..1)")
    ap.add_argument("--beat-subdiv", type=int, default=1,
                    help="拍をさらに分割して判定(1=拍そのまま, 2で拍を二分など)")
    ap.add_argument("--mute-silence", action="store_true",
                    help="ミュートする小節/拍の頭で Common 'silence' を送る")
    ap.add_argument("--phrase-latch", choices=["always","bar","section","enter"], default="enter",
                    help="Common Phraseを送る粒度。enter=休符明けのみ / section=セクション先頭のみ / bar=毎小節 / always=常時")
    ap.add_argument("--mod-preset", default=None)
    ap.add_argument("--mod-variant", choices=["default","low","open"], default=None)
    ap.add_argument("--mod-global-open", type=int, default=None)
    ap.add_argument("--mod-scale", type=float, default=None)
    ap.add_argument("--mod-cap", type=int, default=None)
    ap.add_argument("--mod-hold", choices=["song","section","bar"], default=None)
    args = ap.parse_args()
    is_audio_input = os.path.splitext(args.input)[1].lower() not in [".mid", ".midi"]

    cfg = yaml.safe_load(open(args.config))
    c0, c1 = cfg["ujam"]["c0"], cfg["ujam"]["c1"]
    COMMON = build_common_phrase_map(c0, cfg["phrases_ui_order"])
    STYLE = build_style_phrase_map(c1)
    rules = cfg["rules"]
    fills = cfg["fills"]
    ccconf = cfg["cc"]

    if args.mod_preset is not None:
        ccconf["mod_preset"] = args.mod_preset
    if args.mod_variant is not None:
        ccconf["mod_variant"] = args.mod_variant
    if args.mod_global_open is not None:
        ccconf["mod_global_fallback_open"] = args.mod_global_open
    if args.mod_scale is not None:
        ccconf["mod_scale"] = args.mod_scale
    if args.mod_cap is not None:
        ccconf["mod_cap"] = args.mod_cap
    if args.mod_hold is not None:
        ccconf["mod_hold"] = args.mod_hold

    warned_swing_mod_conflict = False
    warned_unknown_mod_preset = False
    last_mod_val = None
    last_mod_cc = None

    # 入力パース（MIDI or Audio）
    src_notes = []
    tempo = 120.0
    beats = args.beats_per_bar
    if not is_audio_input:
        src = pm.PrettyMIDI(args.input)
        tempos = src.get_tempo_changes()[1]
        if len(tempos):
            tempo = float(tempos[0])
        inst_src = max(src.instruments, key=lambda i: len(i.notes))
        src_notes = sorted(inst_src.notes, key=lambda n: n.start)
        total = max((n.end for n in src_notes), default=0.0)
    else:
        # ---- オーディオ直読み（改良版）----
        dummy, dur, est_tempo, beat_times = audio_onsets_to_dummy_notes(args.input)
        src_notes = sorted(dummy, key=lambda n: n.start)
        tempo = args.force_bpm or (est_tempo if est_tempo > 0 else 120.0)
        dur0 = float(dur) if dur and dur > 0 else 0.0
        last_onset = (src_notes[-1].end if src_notes else 0.0)
        avg_beat = (sum((beat_times[i+1]-beat_times[i]) for i in range(len(beat_times)-1)) /
                    max(1, (len(beat_times)-1))) if (beat_times is not None and len(beat_times) > 1) else (60.0/tempo)
        est_bar = avg_beat * args.beats_per_bar
        last_beat = (beat_times[-1] if (beat_times is not None and len(beat_times)>0) else 0.0)
        auto_total = max(dur0, last_beat + est_bar, last_onset + 2.0 * est_bar)
        total = args.force_seconds or auto_total
        total += float(args.tail_pad or 0.0)  # 末尾に余白
        # 最低保障
        if total < 5.0:
            total = 30.0
    if args.force_bpm:
        tempo = args.force_bpm

    spb = 60.0 / tempo
    default_bar_len = beats * spb

    # ---- 小節列の決定：ビート列があれば優先 ----
    bars_from_beats = []
    try:
        if 'beat_times' in locals() and beat_times is not None and len(beat_times) >= 8:
            bars_from_beats = build_bars_from_beats(beat_times, beats_per_bar=beats, dur=total)
    except Exception:
        bars_from_beats = []

    if bars_from_beats:
        bars = [(i, t0, t1) for i, (t0, t1) in enumerate(bars_from_beats)]
        # 代表値（ログやフォールバック用）
        try:
            import statistics as _st
            default_bar_len = _st.median([t1 - t0 for _, t0, t1 in bars])
        except Exception:
            pass
        using = "beats"
    else:
        bars = list(bars_iter(total, default_bar_len))
        using = "tempo"

    # ここは既存の bars_from_beats / tempo フォールバック決定の直後に追加
    MIN_BARS = args.min_bars  # これ未満なら等間隔にフォールバック
    if len(bars) < MIN_BARS:
        bars = list(bars_iter(total, default_bar_len))
        using = "tempo_fallback"

    print(f"[INFO] audio_dur={total:.2f}s tempo_est={tempo:.1f} bar_len~={default_bar_len:.3f}s bars={len(bars)} mode={using}")

    # セクション検出
    energies = [energy(notes_in_window(src_notes, t0, t1)) for _, t0, t1 in bars]
    sections = detect_sections(energies, rules["section_window_bars"])
    if len(sections) < len(bars) and sections:
        sections += [sections[-1]] * (len(bars) - len(sections))
    elif len(sections) > len(bars):
        sections = sections[:len(bars)]
    print(f"[INFO] sections={len(sections)}  notes_src={len(src_notes)}")
    if bars:
        print(f"[INFO] bars_end={bars[-1][1]:.2f}s (target_total={total:.2f}s)")

    # 出力MIDI
    out = pm.PrettyMIDI(initial_tempo=tempo)
    drv = pm.Instrument(program=0, name="UJAM Driver")
    out.instruments.append(drv)
    if args.debug_bars:
        dbg = pm.Instrument(program=0, name="BAR MARK")
        out.instruments.append(dbg)
        for _, t0, _ in bars:
            dbg.notes.append(pm.Note(velocity=10, pitch=84, start=t0, end=t0 + 0.02))

    # CCレーン（Instrumentに付与）
    def add_cc(inst: pm.Instrument, time_s: float, cc_num: int, value: int):
        inst.control_changes.append(pm.ControlChange(cc_num, value, time_s))

    # Vocal Synchro
    synchro_evts = load_synchro_csv(args.synchro) if args.synchro else []

    # スタイル選択の起点
    style_base = int(cfg["style_phrases"]["base"])
    step_B = int(cfg["style_phrases"]["step_B"])
    step_Build = int(cfg["style_phrases"]["step_Build"])
    cycle_8bars = bool(cfg["style_phrases"]["cycle_8bars"])

    def style_for_bar(b_idx: int, sect: str) -> str:
        base = style_base
        if sect == "B":
            base += step_B
        if sect == "Build":
            base += step_Build
        if cycle_8bars:
            base += (b_idx // 8) % 12
        while base > 12:
            base -= 12
        if base < 1:
            base = 1
        return f"style_{base}"

    last_root = 64  # 初期ルート（E）、後で折返し
    carry = None  # None or {"start": float, "end": float, "root": int}
    prev_active = False           # enter 用
    was_active_any = False        # 全モード共通で前バーの有無音を覚える
    last_cat = None
    last_sect = None
    last_style = None
    # 生成ループ
    for (b, t0, t1), sect in zip(bars, sections):
        bar_len_i = (t1 - t0) if (t1 > t0) else default_bar_len

        bn = notes_in_window(src_notes, t0, t1)
        bn_for_gate = bn
        if is_audio_input and args.audio_gate_8th > 0:
            gate_len = (bar_len_i / 8.0) * float(args.audio_gate_8th)
            bn_for_gate = []
            for n in bn:
                bn_for_gate.append(pm.Note(velocity=n.velocity, pitch=n.pitch,
                                           start=n.start, end=min(t1, n.start + gate_len)))
        play_regions = []  # [(sa,sb)] or [(sa,sb,root)]
        mute_hits = []
        if args.chord_hold == "bar":
            r = active_ratio(bn_for_gate, t0, t1)
            if r >= args.bar_rest_thresh:
                play_regions.append((t0, t1))
            elif args.mute_silence and "silence" in COMMON:
                mute_hits.append(t0 + 0.001)
        elif args.chord_hold == "beat":
            for wa, wb in beat_windows(t0, t1, beats, args.beat_subdiv):
                r = active_ratio(bn_for_gate, wa, wb)
                if r >= args.beat_rest_thresh:
                    play_regions.append((wa, wb))
                elif args.mute_silence and "silence" in COMMON:
                    mute_hits.append(wa + 0.001)
        else:  # segment
            segs_raw = build_segments_from_notes(bn_for_gate, t0, t1, min_rest=args.min_rest)

            def _snap(t, bar_len, div):
                if not div:
                    return t
                step = bar_len / div
                return round(t / step) * step

            segs = []
            for sa, sb, active in segs_raw:
                L = sb - sa
                if not active and L < max(args.min_rest * 1.5, 0.05):
                    continue
                if active and L < args.min_chord_len:
                    continue
                sa2 = _snap(sa, bar_len_i, args.chord_grid)
                sb2 = _snap(sb, bar_len_i, args.chord_grid)
                if sb2 - sa2 <= 1e-4:
                    continue
                segs.append((max(t0, sa2), min(t1, sb2), active))

            regions = []
            cur = None
            for sa, sb, active in segs:
                if not active:
                    if cur:
                        regions.append(cur)
                        cur = None
                    continue
                root_here = last_root
                if args.segment_root:
                    act = [n for n in bn if n.start < sb and n.end > sa]
                    if act:
                        root_here = detect_root_midi_simple(act)
                if cur and cur["root"] == root_here and sa - cur["end"] <= args.chord_retrigger_gap:
                    cur["end"] = max(cur["end"], sb)
                else:
                    if cur:
                        regions.append(cur)
                    cur = {"start": sa, "end": sb, "root": root_here}
                last_root = root_here
            if cur:
                regions.append(cur)

            for reg in regions:
                sa, sb, root_use = reg["start"], reg["end"], reg["root"]
                if (sb - sa) < args.min_chord_len:
                    sb = min(t1, sa + args.min_chord_len)
                play_regions.append((sa, sb, root_use))

            if args.mute_silence and "silence" in COMMON:
                seg_prev_active = True
                for sa, sb, active in segs:
                    if not active and seg_prev_active:
                        mute_hits.append(max(t0, sa + 0.0005))
                    seg_prev_active = active

        now_active = bool(play_regions)
        cat = None
        if now_active:
            cat = choose_phrase(bn, bar_len_i, rules, sect)
            limits = cfg.get("phrase_limits", {})
            if cat in set(limits.get("blocklist", [])):
                cat = limits.get("fallback", "open_1_8")

        should_trigger = False
        if args.phrase_latch == "always":
            should_trigger = now_active
        elif args.phrase_latch == "bar":
            # alias of "always"（処理がバー単位のため）
            should_trigger = now_active
        elif args.phrase_latch == "section":
            # セクションが変わった時 or 休符からの復帰でリトリガ
            should_trigger = now_active and ((sect != last_sect) or (not was_active_any))
        else:  # "enter"
            should_trigger = (now_active and not prev_active)
            prev_active = now_active

        if now_active:
            if cat != last_cat:
                should_trigger = True
            last_cat = cat

        # フレーズ: Latch条件を満たした時だけ再トリガ
        if should_trigger and now_active and cat is not None:
            drv.notes.append(pm.Note(velocity=100, pitch=COMMON[cat], start=t0 + 1e-4, end=t0 + 0.10))

        # スタイル: 変更があれば常に送る（フレーズLatchと独立）
        sty = style_for_bar(b, sect)
        if (sty in STYLE) and (sty != last_style):
            drv.notes.append(pm.Note(velocity=90, pitch=STYLE[sty], start=t0 + 0.02, end=t0 + 0.10))
        last_style = sty

        last_sect = sect
        was_active_any = now_active

        for thit in sorted(set(round(t * 1000) / 1000 for t in mute_hits)):
            drv.notes.append(
                pm.Note(velocity=90, pitch=COMMON["silence"], start=thit, end=thit + 0.012)
            )

        if args.chord_hold != "segment" and bn:
            last_root = detect_root_midi_simple(bn)

        # Build list of (sa, sb, root) for this bar
        regions_with_root = []
        if args.chord_hold == "segment":
            regions_with_root = list(play_regions)
        else:
            for reg in play_regions:
                if isinstance(reg, tuple) and len(reg) == 2:
                    sa, sb = reg
                    root_here = last_root
                    if args.segment_root:
                        act = [n for n in bn if n.start < sb and n.end > sa]
                        if act:
                            root_here = detect_root_midi_simple(act)
                    regions_with_root.append((sa, sb, root_here))

        # Merge into carry across bars
        for sa, sb, root_use in regions_with_root:
            if carry and carry["root"] == root_use and (sa - carry["end"]) <= args.chord_retrigger_gap:
                carry["end"] = max(carry["end"], sb)
            else:
                if carry:
                    _emit_chord_block(drv, cfg, carry["root"], carry["start"], carry["end"])
                carry = {"start": sa, "end": sb, "root": root_use}

        # —— スイング：CC と microtiming の両対応 ——
        swing_amt = swing_ratio_16th(bn, bar_len_i)  # 0..1
        if swing_amt < float(rules.get("swing_threshold", 0.0)):
            swing_amt = 0.0
        if rules.get("swing_mode", "both") in ("both", "cc") and ccconf["enable"]:
            val = max(0, min(127, int(swing_amt * 127)))
            swing_ccnum = ccconf.get("swing_cc")
            if swing_ccnum is not None:
                conflict = (ccconf.get("mod_enable") and swing_ccnum == ccconf.get("mod_cc", 1))
                if conflict:
                    if not warned_swing_mod_conflict:
                        print(f"[WARN] swing_cc ({swing_ccnum}) and mod_cc are identical. Skipping swing CC sends to avoid conflicts.")
                        warned_swing_mod_conflict = True
                else:
                    add_cc(drv, t0 + 0.01, swing_ccnum, val)

        # --- Mod lane (low Mod/Damp) ---
        mod_ccnum = ccconf.get("mod_cc", 1)
        if ccconf.get("mod_enable") and mod_ccnum is not None:
            hold = ccconf.get("mod_hold", "section")

            # ---- choose map by preset / variant ----
            mv_default = (ccconf.get("mod_values") or {})
            preset_key = ccconf.get("mod_preset", None)
            if preset_key and str(preset_key) in MOD_PRESETS:
                mv = MOD_PRESETS[str(preset_key)]
            else:
                if preset_key and not warned_unknown_mod_preset:
                    print(f"[WARN] Unknown mod_preset '{preset_key}'. Falling back to mod_variant.")
                    warned_unknown_mod_preset = True
                variant = str(ccconf.get("mod_variant", "default")).lower()
                if variant == "low":
                    mv = (ccconf.get("mod_values_low") or mv_default)
                elif variant == "open":
                    mv = (ccconf.get("mod_values_open") or mv_default)
                else:
                    mv = mv_default

            # ---- global fallback (force-open) has highest priority ----
            gfb = ccconf.get("mod_global_fallback_open", None)
            if gfb is not None:
                base_val = int(gfb)
            else:
                base_val = int(mv.get(sect, mv.get("default", 32)))

            # ---- scale & cap ----
            scale = float(ccconf.get("mod_scale", 1.0))
            val = int(round(base_val * scale))
            cap = ccconf.get("mod_cap", None)
            if cap is not None:
                try:
                    val = min(val, int(cap))
                except Exception:
                    pass
            val = max(0, min(127, val))

            # ---- should_send based on hold & activity ----
            should_send = False
            if hold == "song":
                should_send = (b == 0)
            elif hold == "section":
                should_send = (b == 0 or sections[b - 1] != sect)
            else:  # "bar"
                should_send = True
            if ccconf.get("mod_only_when_active") and not play_regions:
                should_send = False

            if should_send:
                # dedup identical sends
                if not (ccconf.get("mod_send_if_changed_only", True) and last_mod_val == val and last_mod_cc == mod_ccnum):
                    offset = float(ccconf.get("mod_offset", 0.015))
                    if ccconf.get("swing_cc") == mod_ccnum:
                        offset = max(offset, 0.015)

                    ramp_ms = int(ccconf.get("mod_ramp_ms", 0) or 0)
                    if ramp_ms > 0 and last_mod_val is not None and last_mod_val != val:
                        # optional tiny ramp: send previous first, then new
                        add_cc(drv, t0 + 0.0, mod_ccnum, max(0, min(127, int(last_mod_val))))
                    add_cc(drv, t0 + offset, mod_ccnum, val)

                    last_mod_val = val
                    last_mod_cc = mod_ccnum

            if ccconf.get("mod_debug"):
                print(f"[MOD] bar={b} sect={sect} val={val} hold={hold} preset={ccconf.get('mod_preset')} variant={ccconf.get('mod_variant')}")

        if rules.get("swing_mode", "both") in ("both", "micro"):
            # コードとフレーズにまとめて適用
            bar_notes = [
                n for n in drv.notes
                if t0 <= n.start < t1 and (n.end - n.start) > 0.03
            ]
            apply_swing_microtiming(bar_notes, bar_len_i, swing_amt)

        # —— フィル辞書（末小節など） ——
        only_if_active = bool(play_regions)
        if fills["enable"] and only_if_active:
            is_section_end = (b + 1 < len(sections) and sections[b] != sections[b + 1]) or (
                b == len(sections) - 1
            )
            is_8end = (b % 8) == 7
            if (fills["build_on_section_end"] and is_section_end) or (
                rules["build_every_8bars"] and is_8end
            ):
                key = (
                    "build_up_muted_1_8" if "build_up_muted_1_8" in COMMON else "build_up_open_1_8"
                )
                drv.notes.append(
                    pm.Note(
                        velocity=110,
                        pitch=COMMON[key],
                        start=t1 - 0.5 * bar_len_i,
                        end=t1 - 0.40 * bar_len_i,
                    )
                )
            # ---- Pick Slide の頻度制御（セクション終端／N小節ごと）----
            if "pick_slide" in COMMON:
                put_pick = False
                if fills.get("pickslide_on_section_end", False) and is_section_end:
                    put_pick = True
                nbar = int(fills.get("pickslide_every_n_bars", 0) or 0)
                if nbar > 0 and ((b + 1) % nbar == 0):
                    put_pick = True
                if fills.get("pickslide_on_last_eighth", False):
                    # 明示ONのときだけ毎小節
                    put_pick = True
                if put_pick:
                    vel = int(fills.get("pickslide_velocity", 110))
                    drv.notes.append(
                        pm.Note(
                            velocity=vel,
                            pitch=COMMON["pick_slide"],
                            start=t1 - bar_len_i / 8.0,
                            end=t1 - bar_len_i / 8.0 + 0.05,
                        )
                    )

        # —— Vocal Synchro: 近傍イベントを拾って装飾 ——
        if synchro_evts:
            win_s = bar_len_i / 8.0  # ±8分の窓
            inbar = [ev for ev in synchro_evts if t0 - win_s <= ev[0] < t1 + win_s]
            for t, typ in inbar:
                tt = nearest_grid_time(t, bar_len_i, div=8)
                if typ == "cadence" and "slide_down" in COMMON:
                    drv.notes.append(
                        pm.Note(
                            velocity=120,
                            pitch=COMMON["slide_down"],
                            start=max(t0, tt - 0.05),
                            end=max(t0, tt),
                        )
                    )
                elif typ == "accent" and "muted_1_16_riding" in COMMON:
                    drv.notes.append(
                        pm.Note(
                            velocity=100,
                            pitch=COMMON["muted_1_16_riding"],
                            start=max(t0, tt),
                            end=min(t1, tt + 0.08),
                        )
                    )

        # —— Dynamics CC（ざっくり：セクションに応じて） ——
        if ccconf["enable"]:
            dyn = {"A": 70, "B": 95, "Build": 115}.get(sect, 90)
            add_cc(drv, t0 + 0.02, ccconf["dyn_cc"], dyn)

    if carry:
        _emit_chord_block(drv, cfg, carry["root"], carry["start"], carry["end"])

    out.write(args.out_mid)
    print("Wrote:", args.out_mid)


if __name__ == "__main__":
    main()
