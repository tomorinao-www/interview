import json
import os
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# 从环境变量中获取 API 密钥
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_proxy = os.getenv("OPENAI_PROXY", "http://127.0.0.1:7890")

def get_store():
    try:
        with open("tmp.json", "r") as f:
            pages = json.load(f)

        text_splitter = CharacterTextSplitter(chunk_size=1500, separator="\n")
        docs, metadatas = [], []
        for page in pages:
            splits = text_splitter.split_text(page["text"])
            docs.extend(splits)
            metadatas.extend([{"source": page["source"]}] * len(splits))

        store = FAISS.from_texts(docs,
                                OpenAIEmbeddings(openai_proxy=openai_proxy,
                                                openai_api_key=openai_api_key), 
                                metadatas=metadatas)
        return store
    except Exception as e:
        print(f"error! {e}")
