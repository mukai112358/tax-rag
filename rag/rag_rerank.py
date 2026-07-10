"""ハイブリッド検索 ＋ リランキングのRAG

候補を多めに取り（wide_hybrid）、リランカーで採点し直して上位3件に絞る。
"""

from langchain_core.output_parsers import StrOutputParser
from rag_common import wide_hybrid, rerank, llm, base_prompt, format_docs


def answer(question: str) -> dict:
    candidates = wide_hybrid.invoke(question)
    top_docs = rerank(question, candidates)
    context = format_docs(top_docs)
    ans = (base_prompt | llm | StrOutputParser()).invoke(
        {"context": context, "question": question}
    )
    return {
        "answer": ans,
        "contexts": [d.page_content for d in top_docs],
        "sources": sorted({d.metadata["source"] for d in top_docs}),
    }


if __name__ == "__main__":
    r = answer("青色申告特別控除はいくらですか？")
    print(r["answer"])
    print("出典:", r["sources"])
