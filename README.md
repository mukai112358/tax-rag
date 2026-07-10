# 税務Q&A RAG — RAG手法の実装と比較

国税庁「タックスアンサー」15記事を知識源に、**RAGの主要手法を一通り実装して横並びで比較**したプロジェクト。RAGを体系的に理解することを目的とする。

- **題材**: フリーランス・個人事業主向けの税務Q&A（青色申告・消費税・インボイス等）
- **知識源**: 国税庁タックスアンサー 15記事（`data/*.txt`）
- **共通の作り**: どの手法も同じ呼び出し方 `answer("質問")` で使えるようそろえてある

## 技術スタック

| 分類 | 使用技術 |
|---|---|
| フレームワーク | LangChain 1.x |
| LLM | Google Gemini（`gemini-3.1-flash-lite`） |
| 埋め込み | `models/gemini-embedding-001` ／ `intfloat/multilingual-e5-small` |
| ベクトルDB | Chroma |
| キーワード検索 | BM25（janome） |
| リランカー | `hotchpotch/japanese-reranker-cross-encoder-small-v1` |
| グラフDB | Neo4j Aura ＋ Cypher |
| 評価 / 観測 | RAGAS / LangSmith |

## 手法一覧

| # | 手法 | 概要 |
|---|---|---|
| ① | ベクトル検索 | 質問と意味が近いチャンクを取得して回答 |
| ② | ハイブリッド検索 | ベクトル＋BM25（単語一致）を統合（`EnsembleRetriever`） |
| ③ | リランキング | 候補を CrossEncoder で採点し直して上位に絞る |
| ④ | CRAG | 取得した文脈を3段階評価し、不十分ならWeb検索で補完（LangGraph） |
| ⑤ | Agentic RAG | LLMがツール（DB/Web検索）を自分で選ぶ（`create_agent`） |
| ⑥ | セマンティック分割 | 意味の切れ目でチャンク化（`SemanticChunker`） |
| ⑦ | VLM ＋ RAG | 画像を読み取り、キーワードを作ってRAGで補強し回答 |
| ⑧ | GraphRAG | 記事を知識グラフ化して Neo4j に保存し、Cypher で2ホップ辿って回答 |
| ⑨ | 文書検索 × GraphRAG | 文書検索（本文）とグラフ検索（関係）を組み合わせて回答 |

## ディレクトリ構成

```
data/                 国税庁タックスアンサー 15記事（前処理で取得）
rag/                  手法ごとのプログラム本体（これが成果物）
  ├ rag_common.py       全手法が共通で使う処理（記事の読み込み・検索・リランク・LLM）
  ├ rag_vector.py       ① ベクトル検索
  ├ rag_hybrid.py       ② ハイブリッド検索
  ├ rag_rerank.py       ③ リランキング
  ├ rag_crag.py         ④ CRAG
  ├ rag_agentic.py      ⑤ Agentic RAG
  ├ rag_semantic.py     ⑥ セマンティック分割
  ├ rag_vlm.py          ⑦ VLM ＋ RAG
  ├ rag_graphrag.py     ⑧ GraphRAG
  └ rag_graph_hybrid.py ⑨ 文書検索 × GraphRAG
notebooks/            各手法を実際に動かして試すノート（01〜08）
evaluate.ipynb        全手法をまとめて評価・比較（RAGAS）
studyRAG.ipynb        学習用ノート（全手法を1ファイルにまとめたもの）
```

`rag/` の各ファイルが1つの手法に対応し、`notebooks/` はそれを動かして確認する場所。
どのファイルも `rag_common.py`（共通処理）を土台に使う。

## セットアップ

`.env` に以下を設定（`.gitignore` 済み）:

```
GOOGLE_API_KEY=...
NEO4J_URI=neo4j+s://...
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=...
```

## 実行

各ノートを上から順に実行するだけ（前処理は `rag_common` の import 時に自動実行）。

```python
import rag_rerank
result = rag_rerank.answer("青色申告特別控除はいくらですか？")
print(result["answer"], result["sources"])
```

- ⑧ GraphRAG は初回のみ Neo4j にグラフを構築（以降は再利用）
- ⑦ VLM は画像を引数に渡す：`answer(question, image_path)`

## 評価

`evaluate.ipynb` で全手法を同じ質問セットに回し、RAGAS（Faithfulness / Context Recall）で比較する。全手法が同じ `answer()` 入口を持つためループで一括採点できる。
