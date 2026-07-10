"""自律型 Agentic RAG（create_agent。LLMがツールを自分で選ぶ）

CRAGと違い、流れ（DBを引くか・Webも使うか・もう答えるか）はLLMが毎回判断する。
"""

from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_agent

from rag_common import wide_hybrid, rerank, llm, format_docs


_search = DuckDuckGoSearchRun()


def _text(message):
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content
                   if isinstance(b, dict) and b.get("type") == "text")


@tool
def db_search(query: str) -> str:
    """国税庁タックスアンサーのDBから税務の参考情報を検索する。
    日本の税金・控除・申告など、制度に関する質問にはまずこれを使う。"""
    candidates = wide_hybrid.invoke(query)
    top_chunks = rerank(query, candidates)
    return format_docs(top_chunks)


@tool
def web_search(query: str) -> str:
    """最新情報や、DBに載っていない一般的な情報をWebから検索する。
    DBで十分な答えが得られなかったときに使う。"""
    return _search.invoke(query)


system_prompt = (
    "あなたは日本の税務に詳しいアシスタントです。"
    "税務の質問にはまず db_search を使ってDBを調べてください。"
    "DBで十分に答えられないと判断したときだけ web_search を使ってください。"
    "十分な情報が集まったら、それを根拠に日本語で分かりやすく答えてください。"
)

agent = create_agent(model=llm, tools=[db_search, web_search], system_prompt=system_prompt)


def answer(question: str) -> dict:
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    messages = result["messages"]

    contexts = [m.content for m in messages if m.__class__.__name__ == "ToolMessage"]
    used_tools = sorted({tc["name"] for m in messages
                         if m.__class__.__name__ == "AIMessage" for tc in m.tool_calls})

    return {
        "answer": _text(messages[-1]),
        "contexts": contexts,
        "sources": used_tools,
    }


if __name__ == "__main__":
    r = answer("青色申告特別控除はいくらですか？")
    print(r["answer"])
    print("使ったツール:", r["sources"])
