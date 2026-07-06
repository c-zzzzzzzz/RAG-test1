from typing import List, Tuple, Optional
import numpy as np
from sentence_transformers import SentenceTransformer


class VectorRetriever:
    """sentence-transformers を使った埋め込みベースの検索器。

    文書とクエリを密ベクトルに変換し、コサイン類似度で検索する。
    multilingual-e5 系モデルに対応するため、クエリには "query: "、
    文書には "passage: " のプレフィックスを付与する。
    """

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-small",
        device: str = "cpu",
    ) -> None:
        """VectorRetriever を初期化する。

        Args:
            model_name: HuggingFace上のsentence-transformersモデル名。
            device: 推論に使用するデバイス（"cpu" / "cuda"）。
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.documents: List[dict] = []
        self.embeddings: Optional[np.ndarray] = None

    def index(self, documents: List[dict]) -> None:
        """文書リストを埋め込みベクトルに変換して保持する。

        Args:
            documents: {"id": str, "title": str, "text": str} 形式の文書リスト。
        """
        self.documents = documents
        texts = ["passage: " + d["text"] for d in documents]
        self.embeddings = self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )

    def search(self, query: str, top_k: int = 50) -> List[Tuple[int, float]]:
        """全インデックスからクエリに類似した文書を検索する。

        Args:
            query: 検索クエリ文字列。
            top_k: 返す上位文書数。

        Returns:
            (文書インデックス, コサイン類似度) のリスト。スコア降順。
        """
        if self.embeddings is None:
            return []
        query_emb = self.model.encode(
            "query: " + query, normalize_embeddings=True
        )
        scores = np.dot(self.embeddings, query_emb)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]

    def rerank(
        self, query: str, candidate_indices: List[int]
    ) -> List[Tuple[int, float]]:
        """指定された候補文書インデックス群に対してクエリとの類似度を計算する。

        Args:
            query: 検索クエリ文字列。
            candidate_indices: スコア計算対象の文書インデックスリスト。

        Returns:
            (文書インデックス, コサイン類似度) のリスト。
            候補の元の順序に対応する（スコア降順とは限らない）。
        """
        if self.embeddings is None or len(candidate_indices) == 0:
            return []
        query_emb = self.model.encode(
            "query: " + query, normalize_embeddings=True
        )
        candidate_embs = self.embeddings[candidate_indices]
        scores = np.dot(candidate_embs, query_emb)
        return [
            (int(idx), float(score))
            for idx, score in zip(candidate_indices, scores)
        ]

    def __len__(self) -> int:
        return len(self.documents)
