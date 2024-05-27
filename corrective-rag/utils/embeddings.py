from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
import tempfile
import os

embedding = GPT4AllEmbeddings(model_name="all-MiniLM-L6-v2.gguf2.f16.gguf")


def load_doc(source, type):
    if type == "url":
        loader = WebBaseLoader(source)
        docs = loader.load()
    elif type == "file":
        if source:
            docs = []
            for file in source:
                file_extension = os.path.splitext(file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_file.write(file.read())
                    temp_file_path = temp_file.name

                loader = None
                if file_extension == ".pdf":
                    loader = PyPDFLoader(temp_file_path)

                if loader:
                    docs.extend(loader.load())
                    os.remove(temp_file_path)
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=100
    )
    all_splits = text_splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents=all_splits, collection_name="wiki", embedding=embedding
    )
    retriever = vectorstore.as_retriever()

    return retriever


retriever = load_doc("https://info.cern.ch/", "url")
