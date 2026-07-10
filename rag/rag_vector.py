"""ベクトル検索だけのRAG（一番シンプル）"""

from langchain_core.output_parsers import StrOutputParser
from rag_common import vector_retriever, llm, base_prompt, format_docs


def answer(question: str) -> dict:
    docs = vector_retriever.invoke(question)
    context = format_docs(docs)
    ans = (base_prompt | llm | StrOutputParser()).invoke(
        {"context": context, "question": question}
    )
    return {
        "answer": ans,
        "contexts": [d.page_content for d in docs],
        "sources": sorted({d.metadata["source"] for d in docs}),
    }


if __name__ == "__main__":
    r = answer("青色申告特別控除はいくらですか？")
    print(r["answer"])
    print("出典:", r["sources"])
