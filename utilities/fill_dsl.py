from typing import TypedDict, List
from dataclasses import dataclass

# Mapping from simple DSL tokens to drum instrument labels

TOKEN_MAP: dict[str, str] = {
    "T1": "tom1",
    "T2": "tom2",
    "T3": "tom3",
    "K": "kick",
    "SN": "snare",
    "S": "snare",
    "CRASH": "crash",
}

RUN_TOKENS = {
    "RUNUP": "up",
    "RUNDOWN": "down",
    "RUN↑": "up",
    "RUN↓": "down",
}


def _parse_run(tok: str) -> tuple[str, int] | None:
    key = tok.upper()
    for base, direction in RUN_TOKENS.items():
        if key.startswith(base):
            rest = key[len(base) :]
            count = 3
            if rest.startswith("X") and rest[1:].isdigit():
                count = int(rest[1:])
            return direction, count
    return None


class DrumEvent(TypedDict):
    instrument: str
    offset: float
    duration: float
    velocity_factor: float


def _expand_run(
    direction: str, length_beats: float, vel_factor: float, count: int = 3
) -> List[DrumEvent]:
    """Return tom run events.

    Parameters
    ----------
    direction : str
        Either ``"up"`` or ``"down"``.
    length_beats : float
        Total run length in beats.
    vel_factor : float
        Base velocity scaling factor.

    Returns
    -------
    list[DrumEvent]
        Expanded events evenly spaced within ``length_beats``.
    """
    base = ["tom1", "tom2", "tom3"] if direction == "up" else ["tom3", "tom2", "tom1"]
    drums = [base[i % 3] for i in range(max(1, count))]
    drums.append("crash" if direction == "up" else "kick")
    length = max(float(length_beats), 0.001)
    step = length / len(drums)
    events: List[DrumEvent] = []
    for idx, inst in enumerate(drums):
        scale = (
            (idx + 1) / len(drums)
            if direction == "up"
            else (len(drums) - idx) / len(drums)
        )
        events.append(
            {
                "instrument": inst,
                "offset": idx * step,
                "duration": step if idx < len(drums) - 1 else 0.25,
                "velocity_factor": float(vel_factor) * scale,
            }
        )
    return events


def parse(
    template: str,
    length_beats: float = 1.0,
    velocity_factor: float = 1.0,
) -> List[DrumEvent]:
    """Parse simple drum fill DSL into event dictionaries.

    Parameters
    ----------
    template : str
        Space separated token string (e.g. ``"T1 T2 RUNUP"``).
        ``RUNUP``/``RUNDOWN`` (or ``RUN↑``/``RUN↓``) expand to an ascending
        or descending tom run. ``RUNUPx6`` specifies six tom hits before the
        final crash (kick for ``RUNDOWNxN``).
    length_beats : float, optional
        Total length in beats. Offsets are distributed evenly across this span.
    velocity_factor : float, optional
        Velocity multiplier applied to all events.

    Returns
    -------
    List[DrumEvent]
        Sorted list of event dictionaries ready for music21 insertion.

    Raises
    ------
    KeyError
        If an unknown token is encountered.
    """
    tokens = [tok for tok in (template or "").split() if tok]
    if not tokens:
        return []

    # determine total event count considering RUN tokens
    total = 0
    parsed_tokens: list[tuple[str, int] | str] = []
    for tok in tokens:
        run_info = _parse_run(tok)
        if run_info:
            direction, count = run_info
            total += count + 1
            parsed_tokens.append((direction, count))
        else:
            key = tok.upper()
            if key not in TOKEN_MAP:
                raise KeyError(key)
            total += 1
            parsed_tokens.append(key)

    length = max(float(length_beats), 0.001)
    step = length / total
    events: List[DrumEvent] = []
    idx = 0
    for token in parsed_tokens:
        if isinstance(token, tuple):
            direction, count = token
            run_events = _expand_run(
                direction, step * (count + 1), velocity_factor, count
            )
            start_offset = idx * step
            for ev in run_events:
                ev["offset"] += start_offset
                events.append(ev)
                idx += 1
            continue
        key = token

        offset = idx * step
        duration = step if idx < total - 1 else 0.25
        events.append(
            {
                "instrument": TOKEN_MAP[key],
                "offset": offset,
                "duration": duration,
                "velocity_factor": float(velocity_factor),
            }
        )
        idx += 1

    return events


class FillDSLParseError(Exception):
    """Raised when the tom-run DSL cannot be parsed."""


@dataclass
class _Hit:
    token: str


@dataclass
class _Rest:
    pass


@dataclass
class _Tie:
    pass


@dataclass
class _Velocity:
    value: float


@dataclass
class _Offset:
    pos: int


@dataclass
class _Repeat:
    nodes: list
    times: int


def _tokenize_dsl(src: str) -> list[str]:
    """Return token list for tom-run DSL."""
    tokens: list[str] = []
    i = 0
    src = src.strip()
    while i < len(src):
        c = src[i]
        if c.isspace():
            i += 1
            continue
        if src.startswith("T1", i) or src.startswith("T2", i) or src.startswith("T3", i):
            tokens.append(src[i : i + 2])
            i += 2
            continue
        if c in ("K", "S"):
            tokens.append(c)
            i += 1
            continue
        if c == "+":
            tokens.append("+")
            i += 1
            continue
        if c == ".":
            tokens.append(".")
            i += 1
            continue
        if c == "(":
            tokens.append("(")
            i += 1
            continue
        if c == ")":
            tokens.append(")")
            i += 1
            continue
        if c == "x":
            j = i + 1
            while j < len(src) and src[j].isdigit():
                j += 1
            if j == i + 1:
                raise FillDSLParseError("Missing repeat count after 'x'")
            tokens.append(src[i:j])
            i = j
            continue
        if c == ">":
            j = i + 1
            while j < len(src) and (src[j].isdigit() or src[j] == "."):
                j += 1
            tokens.append(src[i:j])
            i = j
            continue
        if c == "@":
            j = i + 1
            while j < len(src) and src[j].isdigit():
                j += 1
            if j == i + 1:
                raise FillDSLParseError("Missing digits after '@'")
            tokens.append(src[i:j])
            i = j
            continue
        raise FillDSLParseError(f"Unexpected character '{c}' at position {i}")
    return tokens


def _parse_seq(tokens: list[str], idx: int = 0) -> tuple[list, int]:
    nodes: list = []
    while idx < len(tokens):
        tok = tokens[idx]
        if tok == ")":
            idx += 1
            break
        if tok == "(":
            inner, idx = _parse_seq(tokens, idx + 1)
            if idx < len(tokens) and tokens[idx].startswith("x"):
                times = int(tokens[idx][1:])
                idx += 1
            else:
                times = 1
            nodes.append(_Repeat(inner, times))
            continue
        if tok in {"T1", "T2", "T3", "K", "S"}:
            nodes.append(_Hit(tok))
            idx += 1
            continue
        if tok == ".":
            nodes.append(_Rest())
            idx += 1
            continue
        if tok == "+":
            nodes.append(_Tie())
            idx += 1
            continue
        if tok.startswith(">"):
            nodes.append(_Velocity(float(tok[1:])))
            idx += 1
            continue
        if tok.startswith("@"):
            pos = int(tok[1:])
            if pos < 0 or pos > 63:
                raise FillDSLParseError("Offset out of range")
            nodes.append(_Offset(pos))
            idx += 1
            continue
        raise FillDSLParseError(f"Unknown token '{tok}'")
    return nodes, idx


def _eval_nodes(nodes: list) -> list[DrumEvent]:
    events: list[DrumEvent] = []
    offset_16th = 0
    current_vel = 1.0

    def process(seq: list) -> None:
        nonlocal offset_16th, current_vel
        for n in seq:
            if isinstance(n, _Hit):
                events.append(
                    {
                        "instrument": TOKEN_MAP[n.token],
                        "offset": offset_16th * 0.25,
                        "duration": 0.25,
                        "velocity_factor": current_vel,
                    }
                )
                offset_16th += 1
                current_vel = 1.0
            elif isinstance(n, _Rest):
                offset_16th += 1
            elif isinstance(n, _Tie):
                if not events:
                    raise FillDSLParseError("'+' with no preceding hit")
                events[-1]["duration"] += 0.25
                offset_16th += 1
            elif isinstance(n, _Velocity):
                current_vel = n.value
            elif isinstance(n, _Offset):
                offset_16th = n.pos
            elif isinstance(n, _Repeat):
                for _ in range(n.times):
                    process(n.nodes)
    process(nodes)
    return events


def parse_fill_dsl(template: str) -> list[DrumEvent]:
    """Parse tom-run DSL string into drum events."""
    tokens = _tokenize_dsl(template or "")
    if not tokens:
        return []
    ast, idx = _parse_seq(tokens)
    if idx != len(tokens):
        raise FillDSLParseError("Trailing tokens after parse")
    return _eval_nodes(ast)

