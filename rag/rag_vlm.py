"""VLM ＋ RAG（画像を読み取り、税務DBで補強して回答）

画像から検索キーワードを作り、そのキーワードで税務DBを検索し、
画像と検索結果の両方を根拠に回答する（拡張版）。
"""

import base64
from langchain_core.messages import HumanMessage

from rag_common import wide_hybrid, rerank, format_docs, llm


def _image_block(image_path):
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return {"type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{data}"}}


def _text(message):
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content
                   if isinstance(b, dict) and b.get("type") == "text")


def answer(question: str, image_path: str) -> dict:
    img = _image_block(image_path)

    query_msg = HumanMessage(content=[
        {"type": "text",
         "text": f"次の質問に答えるため、この画像から税務DBを検索するのに適したキーワードを1行で出してください。\n質問: {question}"},
        img,
    ])
    search_query = _text(llm.invoke([query_msg]))

    candidates = wide_hybrid.invoke(search_query)
    top_chunks = rerank(search_query, candidates)
    context = format_docs(top_chunks)

    answer_msg = HumanMessage(content=[
        {"type": "text",
         "text": f"画像の内容と、以下の参考情報（税務DB）を踏まえて質問に答えてください。\n\n参考情報:\n{context}\n\n質問: {question}"},
        img,
    ])
    ans = _text(llm.invoke([answer_msg]))

    return {
        "answer": ans,
        "contexts": [c.page_content for c in top_chunks],
        "sources": sorted({c.metadata["source"] for c in top_chunks}),
    }


if __name__ == "__main__":
    r = answer("この源泉徴収票をもとに、私が受けられる控除について教えて", "sample.png")
    print(r["answer"])
    print("出典:", r["sources"])
