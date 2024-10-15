from __future__ import annotations

from langchain_core.documents import BaseDocumentTransformer
from langchain_community.document_loaders.base import BaseLoader
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores import FAISS
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

from common import TaskType
from config import get_config
from logger import get_logger

recipes_logger = get_logger("recipes")
config = get_config()

class VectorDatabase:
    def __init__(self, persist_director:Path, embedding:Embeddings, loaders:BaseLoader, transformers:BaseDocumentTransformer):
        self._persist_director = persist_director
        self._loaders: BaseLoader = loaders
        self._transformers: BaseDocumentTransformer = transformers
        self._embedding = embedding


    def load(self, force_recreation: bool = False) -> VectorStore:
        if  self._persist_director.exists() and not force_recreation:
            recipes_logger.info(f"find the vector store in {self._persist_director}")
            return FAISS.load_local(self._persist_director ,self._embedding, allow_dangerous_deserialization=True)
        else:
            return self._create()

    def _create(self) -> VectorStore:
        documents = self._loaders.load()
        documents = self._transformers.transform_documents(documents)
        vector_store = FAISS.from_documents(
            documents=list(documents),
            embedding=self._embedding,
        )

        vector_store.save_local(self._persist_director )
        return vector_store



class VectorDatabaseBuilder:

    def __init__(self):
        self._persist_director = ""
        self._loaders: BaseLoader = None
        self._transformers: BaseDocumentTransformer = None
        self._embedding = None

    def with_persist_director(self, persist_director:Path) -> VectorDatabaseBuilder:
        self._persist_director = persist_director
        return self

    def with_embedding(self, embedding:Embeddings) -> VectorDatabaseBuilder:
        self._embedding = embedding
        return self

    def with_transformers(self, transformers: BaseDocumentTransformer) -> VectorDatabaseBuilder:
        self._transformers = transformers
        return self

    def with_loaders(self, loaders: BaseLoader) -> VectorDatabaseBuilder:
        self._loaders = loaders
        return self

    def build(self) -> VectorDatabase:
        return VectorDatabase(self._persist_director, self._embedding, self._loaders, self._transformers)


def create_vector_database(embeddings: Embeddings) -> VectorStore:
    model_configs = config.get_model_configs(TaskType.EMBEDDING)
    loaders = PyPDFDirectoryLoader(config.data_path.recipe_books_path)
    transformers = RecursiveCharacterTextSplitter( chunk_size=1000, chunk_overlap=10,
                                                   length_function=len, add_start_index=True,)


    # The following instructions creates a custom Chroma DB wrapper (you can use your own custom code here instead)
    db_builder = (
        VectorDatabaseBuilder()
        .with_persist_director(model_configs.vector_index_path)
        .with_embedding(embeddings)
        .with_loaders(loaders)
        .with_transformers(transformers)
        .build()
    )
    # Either creates a new Chroma DB or load a pre-existing one
    return db_builder.load()
