"""CRAG（3段階評価 ＋ Web再試行ループ）

DBの検索結果を correct / incorrect / ambiguous の3段階で評価し、
correct→DBだけで回答、incorrect→Webだけ、ambiguous→DB＋Web で回答する。
（流れはこのファイル内の StateGraph で自分で組み立てている）
"""

from typing import TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, START, END

from rag_common import wide_hybrid, rerank, llm, format_docs


search = DuckDuckGoSearchRun()
MAX_RETRIES = 0


db_answer_prompt = ChatPromptTemplate.from_template(
    "以下の参考情報だけを根拠に質問に答えてください \n"
    "情報がなければ、分かりませんと答えてください \n\n"
    "参考情報: {context}\n\n 質問: {question}"
)
web_answer_prompt = ChatPromptTemplate.from_template(
    "以下はWeb検索の結果です。これをもとに質問に分かりやすく答えてください。要約して構いません。 \n\n"
    "Web検索結果: {context}\n\n 質問: {question}"
)
combined_answer_prompt = ChatPromptTemplate.from_template(
    "以下にはDBの参考情報とWeb検索の結果の両方が含まれています。両方を踏まえて質問に答えてください。 \n\n"
    "参考情報: {context}\n\n 質問: {question}"
)
grade_prompt = ChatPromptTemplate.from_template(
    "以下の参考情報は、質問に答えるのにどれくらい役立ちますか？次の3つから1語だけ英語で答えてください。 \n"
    "・十分に答えられる → correct \n"
    "・全く関係ない → incorrect \n"
    "・部分的にしか答えられない → ambiguous\n\n"
    "参考情報: {context}\n\n 質問: {question}"
)
web_grade_prompt = ChatPromptTemplate.from_template(
    "以下はWeb検索の結果です。質問に答えるのに十分ですか？"
    "十分なら「はい」、不十分なら「いいえ」とだけ答えてください。 \n\n"
    "Web検索結果: {context}\n\n 質問: {question}"
)
rewrite_prompt = ChatPromptTemplate.from_template(
    "次の質問を、Web検索でヒットしやすい検索キーワードに言い換えてください。"
    "言い換えた検索語だけを出力してください。 \n"
    "質問: {question}"
)


class State(TypedDict):
    question:     str
    search_query: str
    context:      str
    db_contexts:  list
    web_context:  str
    sources:      list
    grade:        str
    web_ok:       str
    attempts:     int
    answer:       str
    used_contexts: list


def retrieve_node(state):
    candidates = wide_hybrid.invoke(state["question"])
    top_chunks = rerank(state["question"], candidates)
    return {"context": format_docs(top_chunks),
            "db_contexts": [chunk.page_content for chunk in top_chunks],
            "sources": sorted({chunk.metadata["source"] for chunk in top_chunks})}


def grade_node(state):
    result = (grade_prompt | llm | StrOutputParser()).invoke(
        {"context": state["context"], "question": state["question"]}
    ).lower()
    if "incorrect" in result:
        grade = "incorrect"
    elif "ambiguous" in result:
        grade = "ambiguous"
    elif "correct" in result:
        grade = "correct"
    else:
        grade = "ambiguous"
    return {"grade": grade}


def web_search_node(state):
    query = state.get("search_query") or state["question"]
    return {"web_context": search.invoke(query)}


def grade_web_node(state):
    result = (web_grade_prompt | llm | StrOutputParser()).invoke(
        {"context": state["web_context"], "question": state["question"]}
    )
    return {"web_ok": "yes" if "はい" in result else "no"}


def rewrite_node(state):
    new_query = (rewrite_prompt | llm | StrOutputParser()).invoke(
        {"question": state["question"]}
    )
    return {"search_query": new_query.strip(),
            "attempts": state["attempts"] + 1}


def generate_answer_node(state):
    if state["grade"] == "correct":
        context = state["context"]
        sources = state["sources"]
        used_prompt = db_answer_prompt
        used_contexts = state["db_contexts"]

    elif state["grade"] == "incorrect":
        context = state["web_context"]
        sources = ["Web検索"]
        used_prompt = web_answer_prompt
        used_contexts = [state["web_context"]]
        
    else:
        context = state["context"] + "\n\n【Web検索】\n" + state["web_context"]
        sources = state["sources"] + ["Web検索"]
        used_prompt = combined_answer_prompt
        used_contexts = state["db_contexts"] + [state["web_context"]]

    ans = (used_prompt | llm | StrOutputParser()).invoke(
        {"context": context, "question": state["question"]}
    )
    return {"answer": ans, "sources": sources, "used_contexts": used_contexts}


def route_after_grade(state):
    return "generate_answer" if state["grade"] == "correct" else "web_search"


def route_after_web(state):
    if state["web_ok"] == "yes":
        return "generate_answer"
    if state["grade"] == "incorrect" and state["attempts"] < MAX_RETRIES:
        return "rewrite"
    return "generate_answer"


builder = StateGraph(State)
builder.add_node("retrieve", retrieve_node)
builder.add_node("grade", grade_node)
builder.add_node("web_search", web_search_node)
builder.add_node("grade_web", grade_web_node)
builder.add_node("rewrite", rewrite_node)
builder.add_node("generate_answer", generate_answer_node)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "grade")
builder.add_conditional_edges("grade", route_after_grade,
                              {"generate_answer": "generate_answer", "web_search": "web_search"})
builder.add_edge("web_search", "grade_web")
builder.add_conditional_edges("grade_web", route_after_web,
                              {"generate_answer": "generate_answer", "rewrite": "rewrite"})
builder.add_edge("rewrite", "web_search")
builder.add_edge("generate_answer", END)

graph = builder.compile()


def answer(question: str) -> dict:
    result = graph.invoke({"question": question, "attempts": 0})
    return {
        "answer": result["answer"],
        "contexts": result.get("used_contexts", []),
        "sources": result["sources"],
    }


if __name__ == "__main__":
    r = answer("青色申告特別控除はいくらですか？")
    print(r["answer"])
    print("出典:", r["sources"])
