"""文書検索（ハイブリッド＋リランク）× GraphRAG のハイブリッド

文書検索で本文を、GraphRAG で関係を取得し、両方を根拠に回答する。
"""

from langchain_core.output_parsers import StrOutputParser

from rag_common import wide_hybrid, rerank, format_docs, llm, base_prompt
from rag_graphrag import graph_context


def answer(question: str) -> dict:
    candidates = wide_hybrid.invoke(question)
    top_chunks = rerank(question, candidates)
    doc_context = format_docs(top_chunks)

    graph_ctx = graph_context(question)

    context = (
        "【関連する文書】\n" + doc_context +
        "\n【エンティティの関係】\n" + graph_ctx
    )
    ans = (base_prompt | llm | StrOutputParser()).invoke(
        {"context": context, "question": question}
    )
    return {
        "answer": ans,
        "contexts": [c.page_content for c in top_chunks]
                    + [line for line in graph_ctx.split("\n") if line],
        "sources": sorted({c.metadata["source"] for c in top_chunks}),
    }


if __name__ == "__main__":
    r = answer("青色申告特別控除の要件は何ですか")
    print(r["answer"])
    print("出典:", r["sources"])
