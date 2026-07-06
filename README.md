# 日本語ハイブリッドRAG検索システム

SudachiPy + BM25 + sentence-transformers を用いた2段階ハイブリッド検索システム。

## アーキテクチャ

```
クエリ → BM25検索（第1段: 語彙ベース）
            ↓ top-k 候補
         ベクトルリランキング（第2段: 意味ベース）
            ↓
         RRF / 重み付き合算でスコア統合
            ↓ top-n
         LLMコンテキスト文字列
```

## セットアップ

```bash
pip install -r requirements.txt
```

初回実行時に `intfloat/multilingual-e5-small` モデルが自動ダウンロードされます（約250MB）。

## 使用方法

```bash
python main.py "検索クエリ"
python main.py "検索クエリ" rrf       # RRF統合（デフォルト）
python main.py "検索クエリ" weighted  # 重み付き合算
python main.py "検索クエリ" weighted 0.7  # α=0.7
```

### 実行例

```bash
python main.py "東京の観光スポット"
```

## テスト

```bash
pytest tests/ -v
```

## ファイル構成

```
├── README.md
├── requirements.txt
├── data/sample_docs.json       # サンプル文書（10件）
├── src/
│   ├── __init__.py
│   ├── tokenizer.py            # SudachiTokenizer
│   ├── bm25_retriever.py       # BM25Retriever
│   ├── vector_retriever.py     # VectorRetriever
│   ├── rrf_fusion.py           # RRFFusion
│   └── hybrid_searcher.py      # HybridSearcher
├── main.py                     # CLIエントリポイント
└── tests/
    ├── __init__.py
    └── test_search.py          # 5観点のテスト
```

## 責務分割

| クラス | 責務 |
|---|---|
| `SudachiTokenizer` | Sudachiによる形態素解析・正規化形抽出 |
| `BM25Retriever` | BM25による語彙ベース検索 |
| `VectorRetriever` | 密ベクトル埋め込みによる意味検索・リランキング |
| `RRFFusion` | RRF / 重み付き合算によるスコア統合 |
| `HybridSearcher` | 上記4クラスを統括するオーケストレーター |
