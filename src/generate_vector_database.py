from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from PyPDF2 import PdfReader

from common import TaskType
from config import get_config
from logger import get_logger

recipes_logger = get_logger("recipes")
config = get_config()


def data_ingestion(data_path: str) -> list[Document]:
    loader = PyPDFDirectoryLoader(data_path)
    return loader.load_and_split(RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000))


def get_pdf_text(pdf_docs: list[str]) -> str:
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text: str) -> list[str]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)


def generate_vector_store(
    *,
    pdf_path: str,
    pdf_docs: list[str],
    vector_index_path: str,
    embeddings: Embeddings,
) -> None:
    if not pdf_path:
        raw_text = get_pdf_text(pdf_docs)
        text_chunks = get_text_chunks(raw_text)
        metadata = [{"source": f"Document {i}"} for i, chunk in enumerate(text_chunks)]
        vector_store = FAISS.from_texts(
            texts=text_chunks,
            embedding=embeddings,
            metadatas=metadata,
        )
    else:
        documents = data_ingestion(pdf_path)
        vector_store = FAISS.from_documents(
            documents=documents,
            embedding=embeddings,
        )

    vector_store.save_local(vector_index_path)


def list_pdf_files(folder_path: Path) -> list:
    return [str(path) for path in folder_path.glob("*.pdf")]


def get_vector_store(embeddings: Embeddings) -> VectorStore:
    model_configs = config.get_model_configs(TaskType.EMBEDDING)
    vector_index_path = model_configs.vector_index_path
    if not Path(vector_index_path).exists():
        generate_vector_store(
            pdf_path=config.data_path.recipe_books_path,
            pdf_docs=list_pdf_files(Path(config.data_path.recipe_books_path)),
            vector_index_path=vector_index_path,
            embeddings=embeddings,
        )
    else:
        recipes_logger.info(f"find the vector store in {vector_index_path}")
    return FAISS.load_local(
        vector_index_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )
