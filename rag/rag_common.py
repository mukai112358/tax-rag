"""共通モジュール：前処理・DB・検索器・リランク・LLM を1か所にまとめる。

各手法ファイル（rag_vector.py など）は、このモジュールから部品を import して使う。
import された時点で「DBの読み込み（or 初回のみ作成）」と「検索器・リランカーの準備」まで済む。
"""

import os
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from janome.tokenizer import Tokenizer
from sentence_transformers import CrossEncoder


load_dotenv()
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

urls = [
    "https://www.nta.go.jp/taxes/shiraberu/taxanswer/shotoku/2070.htm",
    "https://www.nta.go.jp/taxes/shiraberu/taxanswer/shotoku/2072.htm",
    "https://www.nta.go.jp/taxes/shiraberu/taxanswer/shotoku/2075.htm",
    "https://www.nta.go.jp/taxes/shiraberu/taxanswer/shotoku/2090.htm",
    "https://www.nta.go.jp/taxes/shiraberu/taxanswer/shotoku/2100.htm",
    "https://www.nta.go.jp/taxes/shiraberu/taxanswer/shotoku/2210.htm",
    "https://www.nta.go.jp/taxes/shiraberu/taxanswer/shotoku/1350.htm",
    "https://www.nta.go.jp/taxes/shiraberu/taxanswer/shotoku/2020.htm",
    "https://www.nta.go.jp/taxes/shiraberu/taxanswer/shotoku/1100.htm",
    "https://www.nta.go.jp/taxes/shiraberu/taxanswer/shotoku/1130.htm",
    "https://www.nta.go.jp/taxes/shiraberu/taxanswer/shotoku/1199.htm",
    "https://www.nta.go.jp/taxes/shiraberu/taxanswer/shotoku/2260.htm",
    "https://www.nta.go.jp/taxes/shiraberu/taxanswer/shohi/6101.htm",
    "https://www.nta.go.jp/taxes/shiraberu/taxanswer/shohi/6501.htm",
    "https://www.nta.go.jp/taxes/shiraberu/taxanswer/shohi/6498.htm",
]

if not os.path.exists("chroma_db"):
    for url in urls:
        response = requests.get(url)
        response.encoding = response.apparent_encoding
        soup = BeautifulSoup(response.text, "html.parser")
        body = soup.select_one("div.left-content.contents")
        text = body.get_text("\n", strip=True)
        code = url.split("/")[-1].replace(".htm", "")
        with open(f"data/{code}.txt", "w", encoding="utf-8") as f:
            f.write(text)

texts = []
metadatas = []
for path in Path("data").glob("*.txt"):
    texts.append(path.read_text(encoding="utf-8"))
    metadatas.append({"source": path.stem})

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.create_documents(texts, metadatas=metadatas)

if not os.path.exists("chroma_db"):
    db = Chroma.from_documents(chunks, embeddings, persist_directory="chroma_db")
    print("新規作成:", db._collection.count())
else:
    db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    print("既存DBを読み込み:", db._collection.count())

tokenizer = Tokenizer()
def tokenize(text):
    return [t.surface for t in tokenizer.tokenize(text)]


vector_retriever = db.as_retriever(search_kwargs={"k": 3})
bm25_retriever = BM25Retriever.from_documents(chunks, preprocess_func=tokenize, k=3)
hybrid_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever], weights=[0.5, 0.5]
)

wide_vector = db.as_retriever(search_kwargs={"k": 8})
wide_bm25 = BM25Retriever.from_documents(chunks, preprocess_func=tokenize, k=8)
wide_hybrid = EnsembleRetriever(
    retrievers=[wide_vector, wide_bm25], weights=[0.5, 0.5]
)


reranker = CrossEncoder("hotchpotch/japanese-reranker-cross-encoder-small-v1", max_length=512)

def rerank(question, docs, top_n=3):
    pairs = [(question, d.page_content) for d in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [d for d, s in ranked[:top_n]]


llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")

base_prompt = ChatPromptTemplate.from_template(
    "以下の参考情報だけを根拠に質問に答えてください \n"
    "情報がなければ、分かりませんと答えてください \n\n"
    "参考情報: {context}\n\n 質問: {question}"
)

def format_docs(docs):
    """チャンク（Document）のリストを、本文だけつなげた1本の文字列にする。"""
    text = ""
    for d in docs:
        text += d.page_content + "\n\n"
    return text
