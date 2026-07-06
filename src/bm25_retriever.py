from typing import List, Tuple, Optional
import numpy as np
from rank_bm25 import BM25Okapi


class BM25Retriever:
    """BM25アルゴリズムによる語彙ベース検索を行う検索器。

    SudachiTokenizer でトークン化した文書をインデックスし、
    クエリに対してBM25スコアを計算する。
    """

    def __init__(
        self,
        tokenizer: "SudachiTokenizer",  # noqa: F821
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        """BM25Retriever を初期化する。

        Args:
            tokenizer: トークン化に使用する SudachiTokenizer インスタンス。
            k1: BM25のk1パラメータ（語彙の飽和を制御）。
            b: BM25のbパラメータ（文書長の正規化を制御）。
        """
        self.tokenizer = tokenizer
        self.k1 = k1
        self.b = b
        self.documents: List[dict] = []
        self.bm25: Optional[BM25Okapi] = None

    def index(self, documents: List[dict]) -> None:
        """文書リストをインデックスする。

        Args:
            documents: {"id": str, "title": str, "text": str} 形式の文書リスト。
        """
        self.documents = documents
        tokenized_corpus = [self.tokenizer.tokenize(d["text"]) for d in documents]
        self.bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)

    def search(self, query: str, top_k: int = 50) -> List[Tuple[int, float]]:
        """クエリに対してBM25検索を実行する。

        Args:
            query: 検索クエリ文字列。
            top_k: 返す上位文書数。

        Returns:
            (文書インデックス, BM25スコア) のリスト。スコア降順。
        """
        if self.bm25 is None:
            return []
        tokenized_query = self.tokenizer.tokenize(query)
        scores = np.array(self.bm25.get_scores(tokenized_query), dtype=np.float64)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]

    def __len__(self) -> int:
        return len(self.documents)
