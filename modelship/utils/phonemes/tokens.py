"""Phoneme-token table shared by sherpa-onnx TTS frontends."""


class TokenTable:
    def __init__(self, symbol_to_id: dict[str, int]):
        self.symbol_to_id = symbol_to_id
        self.id_to_symbol = {token_id: symbol for symbol, token_id in symbol_to_id.items()}

    @classmethod
    def load(cls, path: str) -> "TokenTable":
        symbol_to_id: dict[str, int] = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                symbol, _, token_id = line.rstrip("\n").rpartition(" ")
                symbol = symbol or " "
                if len(symbol) != 1:
                    raise ValueError(f"token {symbol!r} in {path} is not a single codepoint")
                symbol_to_id[symbol] = int(token_id)
        return cls(symbol_to_id)

    def __contains__(self, symbol: str) -> bool:
        return symbol in self.symbol_to_id
