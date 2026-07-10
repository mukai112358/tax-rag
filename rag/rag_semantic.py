"""セマンティック分割 ＋ リランキング（固定長版との比較用）
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.output_parsers import StrOutputParser

from rag_common import texts, metadatas, embeddings, tokenize, rerank, llm, base_prompt, format_docs


_split_emb = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")
_splitter = SemanticChunker(_split_emb, sentence_split_regex=r"(?<=[。！？])|\n", min_chunk_size=150)
semantic_chunks = _splitter.create_documents(texts, metadatas=metadatas)

db_semantic = Chroma.from_documents(semantic_chunks, embeddings)

_wide_vector = db_semantic.as_retriever(search_kwargs={"k": 8})
_wide_bm25 = BM25Retriever.from_documents(semantic_chunks, preprocess_func=tokenize, k=8)
_wide_hybrid = EnsembleRetriever(retrievers=[_wide_vector, _wide_bm25], weights=[0.5, 0.5])


def answer(question: str) -> dict:
    candidates = _wide_hybrid.invoke(question)
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
