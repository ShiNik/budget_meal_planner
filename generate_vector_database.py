from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader
from langchain_core.embeddings import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import VectorStore
import os
from common import TaskType
from config import get_config

config = get_config()


def get_pdf_text(pdf_docs) -> str:
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text: str) -> list[str]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def generate_vector_store(
    *, pdf_docs: list[str], vector_index_path: str, embeddings: Embeddings
) -> None:
    raw_text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(raw_text)
    metadata = [
        {"source": f"Document {i}"} for i, chunk in enumerate(text_chunks)
    ]  # Add document links or IDs here
    vector_store = FAISS.from_texts(
        texts=text_chunks, embedding=embeddings, metadatas=metadata
    )
    vector_store.save_local(vector_index_path)


def list_pdf_files(folder_path: str) -> list:
    pdf_files = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_files.append(os.path.join(folder_path, filename))
    return pdf_files


def get_vector_store(embeddings: Embeddings) -> VectorStore:
    model_configs = config.get_model_configs(TaskType.EMBEDDING)
    vector_index_path = model_configs.vector_index_path
    if not os.path.exists(vector_index_path):
        generate_vector_store(
            pdf_docs=list_pdf_files(config.data_path.recipe_books_path),
            vector_index_path=vector_index_path,
            embeddings=embeddings,
        )
    else:
        print(f"find the vector store in {vector_index_path}")
    return FAISS.load_local(
        vector_index_path, embeddings, allow_dangerous_deserialization=True
    )
