from __future__ import annotations

from typing import List, Optional

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from rag.config_rag import RAGConfig


class VectorRetriever:
    """
    Loads an existing persisted Chroma DB and retrieves relevant chunks.
    """

    def __init__(self, cfg: RAGConfig):
        self.cfg = cfg
        self._vector_db: Optional[Chroma] = None

    def load_vector_db(self) -> Chroma:
        """Load the persisted vector database"""
        # Ensure persist directory exists
        if not self.cfg.persist_dir.exists():
            raise FileNotFoundError(
                f"Vector database not found at {self.cfg.persist_dir}\n"
                f"Please run DBIndexer.build() first to create the database."
            )

        print("=== RAG ===")
        print(f"Loading vector database from {self.cfg.persist_dir}...")
        embedding = OpenAIEmbeddings(model=self.cfg.embedding_model)

        self._vector_db = Chroma(
            persist_directory=str(self.cfg.persist_dir),
            collection_name=self.cfg.collection_name,
            embedding_function=embedding,
        )

        count = self._vector_db._collection.count()
        print(f"✅ Loaded database with {count} chunks")

        return self._vector_db

    @property
    def vector_db(self) -> Chroma:
        """Lazy-load the vector database on first access"""
        if self._vector_db is None:
            self.load_vector_db()
        return self._vector_db

    def retrieve(self, query: str, k: int = None) -> List[Document]:
        """
        Retrieve the k most similar chunks for a query.

        Args:
            query: The search query
            k: Number of chunks to retrieve (uses config default if None)

        Returns:
            List of relevant document chunks
        """
        if k is None:
            k = self.cfg.chunk_retrieve_default
        if k <= 0:
            raise ValueError("k must be > 0")

        print(f"Retrieving {k} chunk(s) for query: '{query}'")
        results = self.vector_db.similarity_search(query, k=k)
        print(f"✅ Retrieved {len(results)} chunk(s)")

        return results