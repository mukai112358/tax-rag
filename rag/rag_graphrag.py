"""GraphRAG（Neo4j に知識グラフを保存し、Cypher で多ホップ検索）

記事からエンティティ・関係を抽出して Neo4j に保存し、質問に意味が近いノードを
起点に2ホップ分の関係を集めて、それを根拠に回答する。
"""

from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_huggingface import HuggingFaceEmbeddings

from rag_common import texts, metadatas, llm, base_prompt


graph = Neo4jGraph()

if graph.query("MATCH (n) RETURN count(n) AS c")[0]["c"] == 0:
    docs = [Document(page_content=t, metadata={"source": m["source"]})
            for t, m in zip(texts, metadatas)]
    graph_docs = LLMGraphTransformer(llm=llm).convert_to_graph_documents(docs)
    graph.add_graph_documents(graph_docs, baseEntityLabel=True, include_source=False)

_node_emb = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")
_vector_index = Neo4jVector.from_existing_graph(
    embedding=_node_emb,
    node_label="__Entity__",
    text_node_properties=["id"],
    embedding_node_property="embedding",
    retrieval_query="RETURN node.id AS text, score, {} AS metadata",
)


def graph_context(question, k=3):
    ents = [d.page_content for d in _vector_index.similarity_search(question, k=k)]
    rows = graph.query(
        """
        MATCH (n)-[r*1..2]-(m)
        WHERE n.id IN $ents
        UNWIND r AS rel
        RETURN DISTINCT startNode(rel).id AS src, type(rel) AS rel, endNode(rel).id AS tgt
        """,
        params={"ents": ents},
    )
    return "\n".join(f"{row['src']} -[{row['rel']}]-> {row['tgt']}" for row in rows)


def answer(question: str) -> dict:
    context = graph_context(question)
    ans = (base_prompt | llm | StrOutputParser()).invoke(
        {"context": context, "question": question}
    )
    return {
        "answer": ans,
        "contexts": [line for line in context.split("\n") if line],
        "sources": [],
    }


if __name__ == "__main__":
    r = answer("青色申告特別控除の要件は何ですか")
    print(r["answer"])
    print("出典:", r["sources"])
