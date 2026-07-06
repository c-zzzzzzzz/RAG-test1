from typing import List, Tuple, Optional
from .tokenizer import SudachiTokenizer
from .bm25_retriever import BM25Retriever
from .vector_retriever import VectorRetriever
from .rrf_fusion import RRFFusion


class HybridSearcher:
    """2段階ハイブリッドRAG検索システムのメインオーケストレーター。

    1. BM25で語彙ベース検索（高速に候補を絞り込み）
    2. ベクトル類似度でリランキング
    3. RRF / 重み付き合算でスコア統合
    4. LLMコンテキスト文字列の生成
    """

    def __init__(
        self,
        tokenizer: SudachiTokenizer,
        bm25_retriever: BM25Retriever,
        vector_retriever: VectorRetriever,
        fusion: RRFFusion,
    ) -> None:
        """HybridSearcher を初期化する。

        Args:
            tokenizer: トークン化器。
            bm25_retriever: BM25検索器。
            vector_retriever: ベクトル検索器。
            fusion: スコア統合器。
        """
        self.tokenizer = tokenizer
        self.bm25 = bm25_retriever
        self.vector = vector_retriever
        self.fusion = fusion

    def index(self, documents: List[dict]) -> None:
        """文書リストを両方の検索器にインデックスする。

        Args:
            documents: {"id": str, "title": str, "text": str} 形式の文書リスト。
        """
        self.bm25.index(documents)
        self.vector.index(documents)

    def search(
        self,
        query: str,
        top_k_1st: int = 50,
        top_n: int = 5,
    ) -> List[Tuple[int, float]]:
        """ハイブリッド検索を実行する。

        1. BM25で top_k_1st 件の候補を取得
        2. 候補に対してベクトル類似度スコアを計算
        3. スコア統合し上位 top_n 件を返す

        Args:
            query: 検索クエリ。
            top_k_1st: BM25の上位取得件数。
            top_n: 最終的に返す件数。

        Returns:
            (文書インデックス, 統合スコア) のリスト。スコア降順。
        """
        bm25_results = self.bm25.search(query, top_k=top_k_1st)
        if not bm25_results:
            return []
        candidate_indices = [idx for idx, _ in bm25_results]
        vector_results = self.vector.rerank(query, candidate_indices)
        return self.fusion.fuse(bm25_results, vector_results, top_n=top_n)

    def format_context(
        self, query: str, top_n: int = 5
    ) -> str:
        """検索結果をLLMコンテキスト文字列に整形する。

        Args:
            query: 検索クエリ。
            top_n: コンテキストに含める文書数。

        Returns:
            LLMに渡すコンテキスト文字列。
        """
        results = self.search(query, top_n=top_n)
        if not results:
            return "該当する文書が見つかりませんでした。"
        parts: List[str] = []
        for i, (idx, score) in enumerate(results, 1):
            doc = self.bm25.documents[idx]
            parts.append(
                f"--- 結果 {i} (スコア: {score:.4f}) ---\n"
                f"タイトル: {doc['title']}\n"
                f"{doc['text']}\n"
            )
        return "\n".join(parts)
