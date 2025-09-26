#!/usr/bin/env python3
from __future__ import annotations
import argparse
import csv
import fnmatch
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator, Sequence

# -----------------------------
# helpers
# -----------------------------

def human_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    units = ["KB", "MB", "GB", "TB", "PB"]
    x = float(n)
    for u in units:
        x /= 1024.0
        if x < 1024.0:
            return f"{x:.1f} {u}"
    return f"{x:.1f} EB"

@dataclass
class Node:
    path: Path
    is_dir: bool
    size: int = 0
    mtime: float = 0.0
    children: list["Node"] | None = None


def build_tree(root: Path) -> tuple[dict | None, dict[str, int]]:
    if not root.exists():
        return None, {"files": 0, "dirs": 0, "size": 0}

    def _build(path: Path) -> tuple[dict, tuple[int, int, int]]:
        is_dir = path.is_dir()
        node: dict = {
            "name": path.name or str(path),
            "path": str(path),
            "type": "dir" if is_dir else "file",
        }
        if is_dir:
            children = []
            files = 0
            dirs = 1
            size = 0
            for child in sorted(path.iterdir(), key=lambda p: p.name.lower()):
                child_node, (c_files, c_dirs, c_size) = _build(child)
                children.append(child_node)
                files += c_files
                dirs += c_dirs
                size += c_size
            node["children"] = children
            return node, (files, dirs, size)
        else:
            try:
                stat = path.stat()
                size = int(stat.st_size)
            except OSError:
                size = 0
            node["size"] = size
            return node, (1, 0, size)

    tree, stats = _build(root)
    files, dirs, size = stats
    return tree, {"files": files, "dirs": dirs, "size": size}


def render_markdown(node: dict) -> str:
    def _fmt(n: dict) -> str:
        label = n.get("name") or n.get("path", "")
        if n.get("type") == "dir":
            return f"{label}/"
        return label

    lines: list[str] = []

    def _rec(n: dict, prefix: str = "", depth: int = 0, is_last: bool = True) -> None:
        if depth == 0:
            lines.append(_fmt(n))
        else:
            branch = "└── " if is_last else "├── "
            lines.append(prefix + branch + _fmt(n))
        children = n.get("children") or []
        if children:
            new_prefix = prefix + ("    " if is_last else "│   ")
            for i, child in enumerate(children):
                _rec(child, new_prefix, depth + 1, i == len(children) - 1)

    _rec(node)
    return "\n".join(lines)

# -----------------------------
# ignore / include / exclude
# -----------------------------

def load_ignore_patterns(files: Sequence[Path]) -> list[str]:
    pats: list[str] = []
    for p in files:
        if p and p.exists():
            for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                pats.append(line)
    return pats

def match_any(path: Path, patterns: Sequence[str]) -> bool:
    s = str(path)
    return any(fnmatch.fnmatch(s, pat) for pat in patterns)

# -----------------------------
# scanner
# -----------------------------

def walk_tree(
    root: Path,
    *,
    max_depth: int | None,
    include: list[str],
    exclude: list[str],
    ignore_globs: list[str],
    limit_per_dir: int | None,
    max_files: int | None,
    follow_symlinks: bool,
) -> Iterator[Node]:
    total_seen = 0

    def _iter(dir_path: Path, depth: int) -> Iterator[Node]:
        nonlocal total_seen
        try:
            entries = sorted(dir_path.iterdir(), key=lambda p: p.name.lower())
        except Exception:
            return
        count_in_dir = 0
        for entry in entries:
            if max_files is not None and total_seen >= max_files:
                return
            if limit_per_dir is not None and count_in_dir >= limit_per_dir:
                return
            if not follow_symlinks and entry.is_symlink():
                continue
            rel = entry
            if include and not any(fnmatch.fnmatch(str(rel), pat) for pat in include):
                # ただし include がディレクトリ名に合致する場合は探索を続ける
                if entry.is_dir():
                    pass
                else:
                    continue
            if exclude and match_any(rel, exclude):
                continue
            if ignore_globs and match_any(rel, ignore_globs):
                continue

            try:
                is_dir = entry.is_dir()
            except Exception:
                continue
            try:
                stat = entry.stat()
                size = stat.st_size
                mtime = stat.st_mtime
            except Exception:
                size = 0
                mtime = 0.0

            node = Node(entry, is_dir=is_dir, size=size, mtime=mtime, children=[] if is_dir else None)
            total_seen += 1
            count_in_dir += 1
            yield node

            if is_dir and (max_depth is None or depth < max_depth):
                for ch in _iter(entry, depth + 1):
                    assert node.children is not None
                    node.children.append(ch)

    root = root.resolve()
    root_node = Node(root, is_dir=True, size=0, mtime=0.0, children=[])
    for n in _iter(root, 1):
        root_node.children.append(n)
    yield root_node

# -----------------------------
# rendering
# -----------------------------

def render_text(node: Node, *, show_sizes: bool, show_mtime: bool, base: Path, max_depth: int | None) -> str:
    lines: list[str] = []
    def fmt(node: Node) -> str:
        parts = [node.path.name or str(node.path)]
        if show_sizes and not node.is_dir:
            parts.append(f"({human_bytes(node.size)})")
        if show_mtime and node.mtime:
            parts.append(datetime.fromtimestamp(node.mtime).strftime("[%Y-%m-%d %H:%M]") )
        return " ".join(parts)

    def _rec(node: Node, prefix: str = "", is_last: bool = True, depth: int = 0):
        branch = "└── " if is_last else "├── "
        label = fmt(node) if depth > 0 else f"{node.path.relative_to(base) if node.path != base else node.path.name}"
        if depth == 0:
            lines.append(str(label))
        else:
            lines.append(prefix + branch + label)
        if node.children and (max_depth is None or depth < max_depth):
            new_prefix = prefix + ("    " if is_last else "│   ")
            for i, ch in enumerate(node.children):
                _rec(ch, new_prefix, i == len(node.children) - 1, depth + 1)
    _rec(node, depth=0)
    return "\n".join(lines)

def collect_stats(node: Node) -> tuple[Counter, dict[Path, tuple[int,int]]]:
    ext_counter: Counter[str] = Counter()
    dir_stats: dict[Path, tuple[int, int]] = defaultdict(lambda: (0, 0))  # (files, bytes)

    def _rec(n: Node):
        if n.is_dir:
            for ch in (n.children or []):
                _rec(ch)
        else:
            ext = n.path.suffix.lower()
            ext_counter[ext] += 1
            files, bytes_ = dir_stats[n.path.parent]
            dir_stats[n.path.parent] = (files + 1, bytes_ + n.size)
    _rec(node)
    return ext_counter, dir_stats

# -----------------------------
# write-out
# -----------------------------

def write_output(nodes: list[Node], *, fmt: str, show_sizes: bool, show_mtime: bool, base: Path,
                 summary: str | None, top_n_large: int | None, out: Path | None):
    buf: list[str] = []

    def emit(s: str = ""):
        buf.append(s)

    # summaries (shared)
    all_files: list[tuple[Path,int]] = []
    def _collect(nn: Node):
        if nn.is_dir:
            for ch in (nn.children or []):
                _collect(ch)
        else:
            all_files.append((nn.path, nn.size))
    for n in nodes: _collect(n)

    if fmt in {"txt", "md"}:
        for i, n in enumerate(nodes):
            try:
                rel = n.path.relative_to(base)
            except Exception:
                try:
                    rel = n.path.relative_to(Path(base).resolve())
                except Exception:
                    rel = n.path
            if rel == base:
                title_path = rel.name if hasattr(rel, 'name') else rel
            else:
                title_path = rel
            fmt_safe = locals().get('fmt', globals().get('fmt', 'md'))
            title = f"TREE @ {title_path} ({fmt_safe})"

            if fmt == "md":
                emit(f"# {title}")
            else:
                emit(title)
            emit("")
            emit(render_text(n, show_sizes=show_sizes, show_mtime=show_mtime, base=base, max_depth=None))
            emit("")

        if summary:
            # aggregate ext or dir
            if summary == "ext":
                counter = Counter()
                total_bytes = 0
                for p, sz in all_files:
                    counter[p.suffix.lower()] += 1
                    total_bytes += sz
                if fmt == "md":
                    emit("**拡張子サマリ**\n")
                    emit("| ext | files | size |\n|---:|---:|---:|")
                    for ext, cnt in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])):
                        size_sum = sum(sz for (pp, sz) in all_files if pp.suffix.lower() == ext)
                        emit(f"| {ext or '(none)'} | {cnt:,} | {human_bytes(size_sum)} |")
                    emit("")
                else:
                    emit("[Summary by extension]")
                    for ext, cnt in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])):
                        size_sum = sum(sz for (pp, sz) in all_files if pp.suffix.lower() == ext)
                        emit(f"  {ext or '(none)'} : files={cnt:,}, size={human_bytes(size_sum)}")
                    emit("")
            elif summary == "dir":
                dir_counter: dict[Path, tuple[int,int]] = defaultdict(lambda: (0,0))
                for p, sz in all_files:
                    f, b = dir_counter[p.parent]
                    dir_counter[p.parent] = (f+1, b+sz)
                rows = sorted(dir_counter.items(), key=lambda kv: (-kv[1][1], str(kv[0]).lower()))
                if fmt == "md":
                    emit("**ディレクトリ別サマリ**\n")
                    emit("| dir | files | size |\n|:--|--:|--:|")
                    for d, (f, b) in rows:
                        emit(f"| {d} | {f:,} | {human_bytes(b)} |")
                    emit("")
                else:
                    emit("[Summary by directory]")
                    for d, (f, b) in rows:
                        emit(f"  {d} : files={f:,}, size={human_bytes(b)}")
                    emit("")

        if top_n_large:
            top = sorted(all_files, key=lambda kv: kv[1], reverse=True)[:top_n_large]
            if fmt == "md":
                emit(f"**Largest {top_n_large} files**\n")
                emit("| size | path |\n|--:|:--|")
                for p, sz in top:
                    emit(f"| {human_bytes(sz)} | `{p}` |")
                emit("")
            else:
                emit(f"[Largest {top_n_large} files]")
                for p, sz in top:
                    emit(f"  {human_bytes(sz):>10}  {p}")
                emit("")

        out_text = "\n".join(buf)
        if out:
            out.write_text(out_text, encoding="utf-8")
        else:
            print(out_text)
        return

    if fmt == "csv":
        # flat listing
        rows: list[tuple[str,str,int,str,str]] = []
        for p, sz in all_files:
            try:
                mtime = datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")
            except Exception:
                mtime = ""
            rows.append((str(p), "file", sz, p.suffix.lower(), mtime))
        if out:
            with out.open("w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["path","type","size","ext","mtime"])
                w.writerows(rows)
        else:
            w = csv.writer(sys.stdout)
            w.writerow(["path","type","size","ext","mtime"])
            w.writerows(rows)
        return

    if fmt == "json":
        def to_dict(n: Node) -> dict:
            d = {
                "name": n.path.name,
                "path": str(n.path),
                "type": "dir" if n.is_dir else "file",
                "size": n.size,
            }
            if n.children is not None:
                d["children"] = [to_dict(ch) for ch in n.children]
            return d
        data = [to_dict(n) for n in nodes]
        out_text = json.dumps(data, ensure_ascii=False, indent=2)
        if out:
            out.write_text(out_text, encoding="utf-8")
        else:
            print(out_text)
        return

# -----------------------------
# CLI
# -----------------------------

def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Repository tree & stats dumper")
    ap.add_argument("root", nargs="?", default=".", help="scan root (default: .)")
    ap.add_argument("--split-root", nargs="*", default=None, help="scan multiple roots and aggregate")
    ap.add_argument("--format", choices=["txt","md","csv","json"], default="txt")
    ap.add_argument("--out", type=str, default=None, help="output file path")
    ap.add_argument("--max-depth", type=int, default=None)
    ap.add_argument("--limit-per-dir", type=int, default=None, help="files per directory limit")
    ap.add_argument("--max-files", type=int, default=None, help="global cap to avoid huge scans")
    ap.add_argument("--include", action="append", default=[], help="glob to include (repeatable)")
    ap.add_argument("--exclude", action="append", default=[], help="glob to exclude (repeatable)")
    ap.add_argument("--ignore-file", action="append", default=[], help="ignore patterns file(s), e.g., .gitignore")
    ap.add_argument("--sizes", action="store_true", help="show human-readable file sizes")
    ap.add_argument("--mtime", action="store_true", help="show modified time")
    ap.add_argument("--summary", choices=["ext","dir"], default=None)
    ap.add_argument("--top-n-large", type=int, default=None)
    ap.add_argument("--follow-symlinks", action="store_true")

    ns = ap.parse_args(argv)

    bases: list[Path]
    if ns.split_root:
        bases = [Path(p) for p in ns.split_root]
    else:
        bases = [Path(ns.root)]

    ignore_globs = load_ignore_patterns([Path(p) for p in ns.ignore_file])

    all_nodes: list[Node] = []
    for base in bases:
        nodes = list(walk_tree(
            base,
            max_depth=ns.max_depth,
            include=ns.include,
            exclude=ns.exclude,
            ignore_globs=ignore_globs,
            limit_per_dir=ns.limit_per_dir,
            max_files=ns.max_files,
            follow_symlinks=ns.follow_symlinks,
        ))
        all_nodes.extend(nodes)

    out_path = Path(ns.out) if ns.out else None
    base_path = Path(ns.root).resolve()
    write_output(all_nodes, fmt=ns.format, show_sizes=ns.sizes, show_mtime=ns.mtime,
                 base=base_path, summary=ns.summary, top_n_large=ns.top_n_large, out=out_path)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
