from typing import List
from sudachipy import dictionary
from sudachipy import tokenizer as sudachi_tokenizer


class SudachiTokenizer:
    """Sudachiを使った日本語形態素解析器。

    分割モード（A/B/C）を設定可能で、正規化形（normalized_form）を
    返すことで表記ゆれを吸収する。
    """

    def __init__(self, split_mode: str = "A") -> None:
        """SudachiTokenizer を初期化する。

        Args:
            split_mode: 分割モード。"A"（短単位）, "B"（中単位）, "C"（長単位）。
                        デフォルトは "A"。
        """
        modes = {
            "A": sudachi_tokenizer.Tokenizer.SplitMode.A,
            "B": sudachi_tokenizer.Tokenizer.SplitMode.B,
            "C": sudachi_tokenizer.Tokenizer.SplitMode.C,
        }
        self._mode = modes.get(split_mode.upper(), sudachi_tokenizer.Tokenizer.SplitMode.A)
        self._tokenizer = dictionary.Dictionary().create()

    def tokenize(self, text: str) -> List[str]:
        """テキストを形態素解析し、各形態素の正規化形を返す。

        Args:
            text: 解析対象の日本語テキスト。

        Returns:
            正規化形（normalized_form）のリスト。
        """
        morphemes = self._tokenizer.tokenize(text, self._mode)
        return [m.normalized_form() for m in morphemes]
