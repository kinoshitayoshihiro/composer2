from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
import fnmatch
from typing import Any, Dict, Iterable, Optional

try:
    from colorama import Fore, Style, init as colorama_init
except ImportError:  # pragma: no cover - optional
    colorama_init = lambda *args, **kw: None  # type: ignore
    class Fore:
        BLUE = ""
        WHITE = ""
        CYAN = ""
    class Style:
        RESET_ALL = ""

__version__ = "0.2.1"


def format_bytes(num: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    for unit in units:
        if num < 1024:
            return f"{num:.1f} {unit}" if unit != "B" else f"{num} B"
        num /= 1024
    return f"{num:.1f} PiB"

@dataclass
class Summary:
    dirs: int = 0
    files: int = 0
    size: int = 0

    def add(self, other: "Summary") -> None:
        self.dirs += other.dirs
        self.files += other.files
        self.size += other.size


def _matches(path: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(path, pat) for pat in patterns)


def build_tree(
    root: Path,
    *,
    include_hidden: bool = False,
    depth: Optional[int] = None,
    match: Iterable[str] | None = None,
    exclude: Iterable[str] | None = None,
    max_files_per_dir: Optional[int] = None,
    follow_symlinks: bool = False,
    visited_real: Optional[set[str]] = None,
    progress: Any = None,
    _rel: str = "",
) -> tuple[dict | None, Summary]:
    match = list(match or [])
    exclude = list(exclude or [])
    summary = Summary()
    if progress:
        progress.update(1)

    name = root.name
    if not include_hidden and name.startswith(".") and _rel:
        return None, summary
    if _rel and _matches(_rel, exclude):
        return None, summary

    rel_path = _rel or name
    is_match = not match or _matches(rel_path, match)

    if root.is_symlink():
        target = os.readlink(root)
        if root.is_dir() and follow_symlinks:
            real = str(root.resolve())
            visited_real = visited_real or set()
            if real in visited_real:
                if not is_match:
                    return None, summary
                summary.files += 1
                return {"type": "symlink", "size": 0, "target": target}, summary
            visited_real.add(real)
            # Recurse into the real directory but keep symlink marker
            node, sum_child = build_tree(
                root.resolve(),
                include_hidden=include_hidden,
                depth=depth,
                match=match,
                exclude=exclude,
                max_files_per_dir=max_files_per_dir,
                follow_symlinks=follow_symlinks,
                visited_real=visited_real,
                progress=progress,
                _rel=rel_path,
            )
            summary.add(sum_child)
            if node is None:
                node = {"type": "dir", "size": 0, "children": {}}
            node["symlink"] = target
            return node, summary
        else:
            if not is_match:
                return None, summary
            summary.files += 1
            return {"type": "symlink", "size": 0, "target": target}, summary

    if root.is_file():
        if not is_match:
            return None, summary
        size = root.stat().st_size
        summary.files += 1
        summary.size += size
        return {"type": "file", "size": size}, summary

    if root.is_dir():
        if depth is not None and depth < 0:
            return None, summary
        summary.dirs += 1
        if depth is not None and depth == 0:
            return {"type": "dir", "size": 0, "children": {}}, summary
        items = sorted(root.iterdir(), key=lambda p: p.name)
        extra = 0
        if max_files_per_dir is not None and len(items) > max_files_per_dir:
            extra = len(items) - max_files_per_dir
            items = items[:max_files_per_dir]
        children: Dict[str, Any] = {}
        size_acc = 0
        for child in items:
            child_rel = f"{rel_path}/{child.name}" if rel_path else child.name
            node, sum_child = build_tree(
                child,
                include_hidden=include_hidden,
                depth=None if depth is None else depth - 1,
                match=match,
                exclude=exclude,
                max_files_per_dir=max_files_per_dir,
                follow_symlinks=follow_symlinks,
                visited_real=visited_real,
                progress=progress,
                _rel=child_rel,
            )
            summary.add(sum_child)
            if node is not None:
                children[child.name] = node
                size_acc += node.get("size", 0)
        if extra:
            children["…"] = {"type": "placeholder", "note": f"(+{extra} more)"}
        if not is_match and not children:
            return None, summary
        return {"type": "dir", "size": size_acc, "children": children}, summary
    return None, summary


def render_text(node: dict, prefix: str = "", human_size: bool = False) -> str:
    lines: list[str] = []

    def _walk(n: dict, name: str, pref: str, last: bool) -> None:
        connector = "└── " if last else "├── "
        line = pref + connector + name
        if n["type"] == "symlink":
            line += f" -> {n['target']}"
        if human_size and n["type"] == "dir":
            line += f" ({format_bytes(n['size'])})"
        lines.append(line)
        if n.get("children"):
            new_pref = pref + ("    " if last else "│   ")
            items = sorted(
                n["children"].items(), key=lambda kv: (kv[1]["type"] != "dir", kv[0])
            )
            for i, (child_name, child_node) in enumerate(items):
                label = child_name
                if child_node["type"] == "placeholder":
                    label = child_node["note"]
                _walk(child_node, label, new_pref, i == len(items) - 1)

    _walk(node, prefix or Path.cwd().name, "", True)
    return "\n".join(lines)


def render_markdown(node: dict) -> str:
    return "```\n" + render_text(node) + "\n```\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("root", nargs="?", default=".")
    ap.add_argument("-a", "--include-hidden", action="store_true", help="include hidden files")
    ap.add_argument("-d", "--depth", type=int, default=None, help="depth limit; None=unlimited")
    ap.add_argument("--out", type=str, help="output file path")
    ap.add_argument("--summary", action="store_true", help="append summary statistics")
    ap.add_argument("--max-files-per-dir", type=int, default=None, help="truncate large directories")
    ap.add_argument("--match", action="append", default=[], help="glob to include")
    ap.add_argument("--exclude", action="append", default=[], help="glob to exclude")
    ap.add_argument("--largest", type=int, default=0, help="show top N largest files and dirs")
    ap.add_argument("--progress", action="store_true", help="show progress bar")
    ap.add_argument("--follow-symlinks", action="store_true", help="recurse into symlinked dirs")
    ap.add_argument("--human-size", action="store_true", help="display human-readable sizes")
    ap.add_argument("--color", action="store_true", help="colorize stdout")
    ap.add_argument("--version", action="version", version=f"dump_tree {__version__}")

    ns = ap.parse_args(argv)
    root_path = Path(ns.root)
    if not root_path.exists():
        print(f"Root path not found: {root_path}", file=sys.stderr)
        return 1

    depth = None if ns.depth is not None and ns.depth < 0 else ns.depth

    prog = None
    if ns.progress and sys.stdout.isatty():
        from tqdm import tqdm  # type: ignore
        prog = tqdm(unit="files", desc="scanning")

    node, summary = build_tree(
        root_path,
        include_hidden=ns.include_hidden,
        depth=depth,
        match=ns.match,
        exclude=ns.exclude,
        max_files_per_dir=ns.max_files_per_dir,
        follow_symlinks=ns.follow_symlinks,
        progress=prog,
        _rel="",
    )
    if node is None:
        node = {"type": "dir", "size": 0, "children": {}}

    if prog:
        prog.close()

    largest_files: list[tuple[int, str]] = []
    largest_dirs: list[tuple[int, str]] = []
    if ns.largest:
        def collect(n: dict, path: str) -> None:
            if n["type"] == "file":
                largest_files.append((n["size"], path))
            elif n["type"] == "dir":
                largest_dirs.append((n["size"], path))
                for child_name, child_node in n.get("children", {}).items():
                    if child_node["type"] == "placeholder":
                        continue
                    collect(child_node, f"{path}/{child_name}")
        collect(node, root_path.name)
        import heapq
        largest_files = heapq.nlargest(ns.largest, largest_files)
        largest_dirs = heapq.nlargest(ns.largest, largest_dirs)

    size_text = format_bytes(summary.size) if ns.human_size else f"{summary.size} bytes"
    summary_line = f"{summary.dirs} dirs, {summary.files} files, total_size = {size_text}"

    if ns.out:
        out_path = Path(ns.out)
        if out_path.suffix == ".json":
            obj = node
            if ns.summary or ns.largest:
                obj = {"tree": node}
                if ns.summary:
                    obj["summary"] = {
                        "dirs": summary.dirs,
                        "files": summary.files,
                        "total_size": summary.size,
                    }
                if ns.largest:
                    obj["largest"] = {
                        "files": [
                            {"path": p, "size": s} for s, p in largest_files
                        ],
                        "dirs": [
                            {"path": p, "size": s} for s, p in largest_dirs
                        ],
                    }
            out_path.write_text(json.dumps(obj, indent=2))
        elif out_path.suffix == ".md":
            text = render_markdown(node)
            if ns.summary:
                text += "\n" + summary_line + "\n"
            out_path.write_text(text)
        else:
            text = render_text(node, human_size=ns.human_size)
            if ns.summary:
                text += "\n" + summary_line
            if ns.largest:
                text += "\n\nLargest files:\n" + "\n".join(
                    f"{format_bytes(s) if ns.human_size else s} {p}" for s, p in largest_files
                )
                text += "\nLargest dirs:\n" + "\n".join(
                    f"{format_bytes(s) if ns.human_size else s} {p}" for s, p in largest_dirs
                )
            out_path.write_text(text)
    else:
        use_color = ns.color and sys.stdout.isatty()
        if use_color:
            colorama_init(autoreset=True)

        def _color_line(line: str) -> str:
            if not use_color:
                return line
            if " -> " in line:
                color = Fore.CYAN
            elif line.startswith("└") or line.startswith("├"):
                color = Fore.BLUE
            else:
                color = Fore.WHITE
            return color + line + Style.RESET_ALL

        text = render_text(node, human_size=ns.human_size)
        if use_color:
            text = "\n".join(_color_line(l) for l in text.splitlines())
        print(text)
        if ns.summary:
            print(summary_line)
        if ns.largest:
            print("\nLargest files:")
            for size, path in largest_files:
                print(f"{format_bytes(size) if ns.human_size else size} {path}")
            print("Largest dirs:")
            for size, path in largest_dirs:
                print(f"{format_bytes(size) if ns.human_size else size} {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
