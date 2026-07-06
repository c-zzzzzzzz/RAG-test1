import pytest
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tokenizer import SudachiTokenizer
from src.bm25_retriever import BM25Retriever
from src.vector_retriever import VectorRetriever
from src.rrf_fusion import RRFFusion
from src.hybrid_searcher import HybridSearcher


SAMPLE_DOCS_PATH = Path(__file__).parent.parent / "data" / "sample_docs.json"


def load_docs() -> list[dict]:
    with open(SAMPLE_DOCS_PATH, encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# 1. Sudachi トークン化テスト
# ============================================================

class TestTokenizer:
    def test_normalized_form_katakana_long_vowel(self) -> None:
        """「サーバー」と「サーバ」が同じ正規化形になることを確認する。"""
        tokenizer = SudachiTokenizer()
        tokens_a = set(tokenizer.tokenize("サーバー"))
        tokens_b = set(tokenizer.tokenize("サーバ"))
        common = tokens_a & tokens_b
        assert len(common) > 0, (
            f"共通する正規化形がありません。\n"
            f"  'サーバー' → {tokens_a}\n"
            f"  'サーバ'   → {tokens_b}"
        )

    def test_tokenize_returns_list(self) -> None:
        """tokenize が空でないリストを返すことを確認する。"""
        tokenizer = SudachiTokenizer()
        tokens = tokenizer.tokenize("東京タワーで夜景を見る")
        assert isinstance(tokens, list)
        assert len(tokens) > 0

    def test_split_mode_a_vs_c(self) -> None:
        """分割モードA（短単位）とC（長単位）で結果が異なることを確認する。"""
        text = "東京都港区"
        tok_a = SudachiTokenizer(split_mode="A")
        tok_c = SudachiTokenizer(split_mode="C")
        tokens_a = tok_a.tokenize(text)
        tokens_c = tok_c.tokenize(text)
        assert tokens_a != tokens_c


# ============================================================
# 2. BM25 テスト
# ============================================================

class TestBM25Retriever:
    def test_search_finds_named_entity(self) -> None:
        """クエリの固有名詞を含む文書が上位に来ることを確認する。"""
        tokenizer = SudachiTokenizer()
        bm25 = BM25Retriever(tokenizer=tokenizer)
        docs = load_docs()
        bm25.index(docs)
        results = bm25.search("東京タワー", top_k=5)
        assert len(results) > 0
        top_idx, _ = results[0]
        assert docs[top_idx]["id"] == "doc_0", (
            f"期待: doc_0 (東京タワー), 実際: {docs[top_idx]['id']}"
        )

    def test_search_returns_empty_for_no_match(self) -> None:
        """存在しないキーワードで空リストが返ることを確認する。"""
        tokenizer = SudachiTokenizer()
        bm25 = BM25Retriever(tokenizer=tokenizer)
        docs = load_docs()
        bm25.index(docs)
        results = bm25.search("xyzzyXYZ123", top_k=5)
        assert results == []

    def test_search_respects_top_k(self) -> None:
        """top_k に指定した件数だけ結果が返ることを確認する。"""
        tokenizer = SudachiTokenizer()
        bm25 = BM25Retriever(tokenizer=tokenizer)
        docs = load_docs()
        bm25.index(docs)
        results = bm25.search("ラーメン", top_k=3)
        assert len(results) <= 3


# ============================================================
# 3. ベクトル検索テスト
# ============================================================

class TestVectorRetriever:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.tokenizer = SudachiTokenizer()
        self.docs = load_docs()
        self.vector = VectorRetriever()
        self.vector.index(self.docs)

    def test_paraphrase_hit(self) -> None:
        """言い換え表現（「車」のクエリで「自動車」の文書）がヒットすることを確認する。"""
        results = self.vector.search("車", top_k=5)
        top_indices = [idx for idx, _ in results]
        all_titles = [self.docs[i]["title"] for i in top_indices]
        assert "自動車" in " ".join(all_titles) or any(
            "自動車" in self.docs[i]["text"] for i in top_indices
        ), f"「車」の検索結果に自動車関連文書が見つかりません: {all_titles}"

    def test_search_returns_correct_structure(self) -> None:
        """検索結果が (int, float) のリストであることを確認する。"""
        results = self.vector.search("日本酒", top_k=3)
        assert isinstance(results, list)
        assert len(results) > 0
        for idx, score in results:
            assert isinstance(idx, int)
            assert isinstance(score, float)

    def test_rerank_subset(self) -> None:
        """rerank が指定インデックスのみスコアリングすることを確認する。"""
        results = self.vector.rerank("温泉", [0, 7])
        assert len(results) == 2
        assert results[0][0] == 0  # idx 0
        assert results[1][0] == 7  # idx 7


# ============================================================
# 4. RRF テスト
# ============================================================

class TestRRFFusion:
    def test_rrf_known_ranks(self) -> None:
        """既知の順位リストから期待通りの統合スコアが計算されることを確認する。"""
        fusion = RRFFusion(rrf_k=1)
        bm25_results = [(0, 1.0), (1, 0.8), (2, 0.7)]
        vector_results = [(0, 0.95), (2, 0.90)]
        fused = dict(fusion.fuse(bm25_results, vector_results, top_n=3))
        expected_0 = 1.0 / (1 + 0 + 1) + 1.0 / (1 + 0 + 1)
        expected_1 = 1.0 / (1 + 1 + 1)
        expected_2 = 1.0 / (1 + 2 + 1) + 1.0 / (1 + 1 + 1)
        assert abs(fused[0] - expected_0) < 1e-10, f"doc 0: {fused[0]} != {expected_0}"
        assert abs(fused[1] - expected_1) < 1e-10, f"doc 1: {fused[1]} != {expected_1}"
        assert abs(fused[2] - expected_2) < 1e-10, f"doc 2: {fused[2]} != {expected_2}"

    def test_rrf_order_consistency(self) -> None:
        """スコア降順にソートされていることを確認する。"""
        fusion = RRFFusion(rrf_k=60)
        bm25 = [(0, 0.9), (1, 0.8), (2, 0.7), (3, 0.6)]
        vec = [(3, 0.95), (0, 0.85), (1, 0.75), (2, 0.65)]
        fused = fusion.fuse(bm25, vec, top_n=4)
        scores = [s for _, s in fused]
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))

    def test_weighted_fusion(self) -> None:
        """重み付き合算が正しく計算されることを確認する。"""
        fusion = RRFFusion(method="weighted", alpha=0.6)
        bm25 = [(0, 10.0), (1, 5.0)]
        vec = [(0, 0.8), (1, 0.6)]
        fused = dict(fusion.fuse(bm25, vec, top_n=2))
        expected = 0.6 * 1.0 + 0.4 * 0.8
        assert abs(fused[0] - expected) < 1e-10

    def test_empty_inputs(self) -> None:
        """空の入力に対して空リストが返ることを確認する。"""
        fusion = RRFFusion()
        assert fusion.fuse([], [], top_n=5) == []


# ============================================================
# 5. E2E テスト
# ============================================================

class TestHybridSearcher:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        tokenizer = SudachiTokenizer()
        bm25 = BM25Retriever(tokenizer=tokenizer)
        vector = VectorRetriever()
        fusion = RRFFusion()
        self.searcher = HybridSearcher(
            tokenizer=tokenizer,
            bm25_retriever=bm25,
            vector_retriever=vector,
            fusion=fusion,
        )
        self.docs = load_docs()
        self.searcher.index(self.docs)

    def test_search_returns_top_n(self) -> None:
        """HybridSearcher にクエリを投げて top_n 件が返ることを確認する。"""
        results = self.searcher.search("富士山の登山", top_n=3)
        assert len(results) <= 3
        assert len(results) > 0
        for idx, score in results:
            assert isinstance(idx, int)
            assert isinstance(score, float)

    def test_search_default_top_n(self) -> None:
        """デフォルトの top_n (5) で結果が返ることを確認する。"""
        results = self.searcher.search("ラーメン")
        assert 0 < len(results) <= 5

    def test_format_context(self) -> None:
        """format_context が空でない文字列を返すことを確認する。"""
        context = self.searcher.format_context("日本酒", top_n=3)
        assert isinstance(context, str)
        assert len(context) > 0
        assert "結果 1" in context

    def test_search_without_index_returns_empty(self) -> None:
        """インデックス未構築の searcher が空結果を返すことを確認する。"""
        tokenizer = SudachiTokenizer()
        bm25 = BM25Retriever(tokenizer=tokenizer)
        vector = VectorRetriever()
        fusion = RRFFusion()
        empty_searcher = HybridSearcher(
            tokenizer=tokenizer,
            bm25_retriever=bm25,
            vector_retriever=vector,
            fusion=fusion,
        )
        assert empty_searcher.search("test") == []
