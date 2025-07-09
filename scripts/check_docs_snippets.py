import re
import sys
from pathlib import Path
import yaml

CODE_BLOCK_RE = re.compile(r"```(\w+)\n(.*?)```", re.S)

def validate_block(lang: str, code: str, path: Path) -> list[str]:
    errors = []
    if lang.lower() in {"yaml", "yml"}:
        try:
            yaml.safe_load(code)
        except Exception as exc:
            errors.append(f"{path}: YAML error: {exc}")
    return errors


def check_file(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    errors = []
    for lang, code in CODE_BLOCK_RE.findall(text):
        errors.extend(validate_block(lang, code, path))
    return errors


def main() -> None:
    docs = Path("docs")
    all_errors = []
    for md in docs.rglob("*.md"):
        all_errors.extend(check_file(md))
    for err in all_errors:
        print(err, file=sys.stderr)
    if all_errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
