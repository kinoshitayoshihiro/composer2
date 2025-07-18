class SaxTokenizer:
    """Minimal stub tokenizer used for smoke tests."""

    pad_id = 0

    def __init__(self) -> None:
        self.vocab = {0: "<pad>"}

    def encode(self, events: list[dict[str, object]]) -> list[int]:
        return [0 for _ in events]

    def decode(self, ids: list[int]) -> list[dict[str, object]]:
        return [{} for _ in ids]


__all__ = ["SaxTokenizer"]
