from typing import List, Tuple, Dict, Literal
import numpy as np


class RRFFusion:
    """2つの検索結果を統合するフュージョン器。

    以下の2方式をサポート:
    - "rrf": Reciprocal Rank Fusion による順位ベースの統合
    - "weighted": BM25正規化スコアとベクトル類似度の重み付き合算
    """

    def __init__(
        self,
        rrf_k: int = 60,
        method: Literal["rrf", "weighted"] = "rrf",
        alpha: float = 0.5,
    ) -> None:
        """RRFFusion を初期化する。

        Args:
            rrf_k: RRFの定数k。大きいほど順位の差がスコアに与える影響が小さい。
            method: 統合方式。"rrf" または "weighted"。
            alpha: 重み付き合算時のBM25スコアの重み（0.0〜1.0）。
        """
        self.rrf_k = rrf_k
        self.method = method
        self.alpha = alpha

    def fuse(
        self,
        bm25_results: List[Tuple[int, float]],
        vector_results: List[Tuple[int, float]],
        top_n: int = 5,
    ) -> List[Tuple[int, float]]:
        """2つの検索結果リストを統合する。

        Args:
            bm25_results: (文書インデックス, BM25スコア) のリスト（順位順）。
            vector_results: (文書インデックス, ベクトル類似度) のリスト。
            top_n: 返す上位文書数。

        Returns:
            (文書インデックス, 統合スコア) のリスト。スコア降順。
        """
        if not bm25_results and not vector_results:
            return []
        if self.method == "weighted":
            return self._weighted_fuse(bm25_results, vector_results, top_n)
        return self._rrf_fuse(bm25_results, vector_results, top_n)

    def _rrf_fuse(
        self,
        bm25_results: List[Tuple[int, float]],
        vector_results: List[Tuple[int, float]],
        top_n: int,
    ) -> List[Tuple[int, float]]:
        bm25_ranks: Dict[int, int] = {
            idx: rank for rank, (idx, _) in enumerate(bm25_results)
        }
        vector_ranks: Dict[int, int] = {
            idx: rank for rank, (idx, _) in enumerate(vector_results)
        }
        all_indices = set(bm25_ranks.keys()) | set(vector_ranks.keys())
        scores: Dict[int, float] = {}
        for idx in all_indices:
            s = 0.0
            if idx in bm25_ranks:
                s += 1.0 / (self.rrf_k + bm25_ranks[idx] + 1)
            if idx in vector_ranks:
                s += 1.0 / (self.rrf_k + vector_ranks[idx] + 1)
            scores[idx] = s
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:top_n]

    def _weighted_fuse(
        self,
        bm25_results: List[Tuple[int, float]],
        vector_results: List[Tuple[int, float]],
        top_n: int,
    ) -> List[Tuple[int, float]]:
        bm25_dict: Dict[int, float] = {}
        if bm25_results:
            raw = np.array([s for _, s in bm25_results])
            min_s, max_s = float(raw.min()), float(raw.max())
            diff = max_s - min_s
            for idx, s in bm25_results:
                bm25_dict[idx] = (s - min_s) / diff if diff > 0 else 1.0
        vector_dict: Dict[int, float] = {
            idx: max(s, 0.0) for idx, s in vector_results
        }
        all_indices = set(bm25_dict.keys()) | set(vector_dict.keys())
        scores: Dict[int, float] = {}
        for idx in all_indices:
            b_score = bm25_dict.get(idx, 0.0)
            v_score = vector_dict.get(idx, 0.0)
            scores[idx] = self.alpha * b_score + (1.0 - self.alpha) * v_score
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:top_n]
