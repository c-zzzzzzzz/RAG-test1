import sys
import json
import os
from pathlib import Path

from src.tokenizer import SudachiTokenizer
from src.bm25_retriever import BM25Retriever
from src.vector_retriever import VectorRetriever
from src.rrf_fusion import RRFFusion
from src.hybrid_searcher import HybridSearcher


def load_sample_docs() -> list[dict]:
    data_path = Path(__file__).parent / "data" / "sample_docs.json"
    with open(data_path, encoding="utf-8") as f:
        return json.load(f)


def build_searcher(method: str = "rrf", alpha: float = 0.5) -> HybridSearcher:
    tokenizer = SudachiTokenizer(split_mode="A")
    bm25 = BM25Retriever(tokenizer=tokenizer)
    vector = VectorRetriever()
    fusion = RRFFusion(method=method, alpha=alpha)
    searcher = HybridSearcher(
        tokenizer=tokenizer,
        bm25_retriever=bm25,
        vector_retriever=vector,
        fusion=fusion,
    )
    docs = load_sample_docs()
    print(f"インデックス構築中... ({len(docs)} 文書)", file=sys.stderr)
    searcher.index(docs)
    return searcher


def main() -> None:
    if len(sys.argv) < 2:
        print("使用方法: python main.py <検索クエリ>", file=sys.stderr)
        print("  例:  python main.py '東京の観光スポット'", file=sys.stderr)
        sys.exit(1)

    query = sys.argv[1]
    method = sys.argv[2] if len(sys.argv) > 2 else "rrf"
    alpha = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5

    searcher = build_searcher(method=method, alpha=alpha)
    context = searcher.format_context(query, top_n=5)
    print(context)


if __name__ == "__main__":
    main()
