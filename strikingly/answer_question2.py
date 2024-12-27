import json
import os
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain.chains import ConversationalRetrievalChain
import argparse

parser = argparse.ArgumentParser(description="Paepper.com Q&A")
parser.add_argument("question", type=str, help="Your question for Paepper.com")
args = parser.parse_args()

# 从环境变量中获取 API 密钥
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_proxy = os.getenv("OPENAI_PROXY", "http://127.0.0.1:7890")


def get_store():
    with open("tmp.json", "r") as f:
        pages = json.load(f)

    text_splitter = CharacterTextSplitter(chunk_size=1500, separator="\n")
    docs, metadatas = [], []
    for page in pages:
        splits = text_splitter.split_text(page["text"])
        docs.extend(splits)
        metadatas.extend([{"source": page["source"]}] * len(splits))

    store = FAISS.from_texts(
        docs,
        OpenAIEmbeddings(openai_proxy=openai_proxy, openai_api_key=openai_api_key),
        metadatas=metadatas,
    )
    return store


store = get_store()


vectordbkwargs = {"search_distance": 0.9}
retriever = store.as_retriever(search_kwargs={"search_distance": 0.9})
qa = ConversationalRetrievalChain.from_llm(
    llm=OpenAI(temperature=0),
    retriever=retriever,
    return_source_documents=True,
)

chat_history = []  # no need
result = qa(
    {
        "question": args.question,
        "chat_history": chat_history,
        "vectordbkwargs": vectordbkwargs,
    }
)

print(f"Answer: {result['answer']}")
# document = Document(
#     page_content="Hello, world!", metadata={"source": "https://example.com"}
# )
sources = [doc.metadata["source"] for doc in result["source_documents"]]
print(f"Sources: {sources}")
