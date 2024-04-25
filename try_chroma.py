from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_core.documents.base import Document

# Load
url = "https://lilianweng.github.io/posts/2023-06-23-agent/"
loader = WebBaseLoader(url)
docs = loader.load()

# Split
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, chunk_overlap=100
)
all_splits = text_splitter.split_documents(docs)

embedding = GPT4AllEmbeddings()

db_path = "./db_path"  # Path to SQLite database
chroma = Chroma(persist_directory=db_path, embedding_function=embedding, collection_name="rag-chroma")


# id = chroma.add_documents(documents=all_splits)
# print(id)

retriever = chroma.as_retriever()

query = "What is the agent in reinforcement learning?"
retrieved_docs = retriever.get_relevant_documents(query, top_k=5)

for doc in retrieved_docs:
    print(doc.page_content)
    # print(doc.metadata)
    print("\n\n\n----\n\n\n")
