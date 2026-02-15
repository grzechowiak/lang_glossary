from __future__ import annotations

from typing import List
import shutil

from langchain_core.documents import Document
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from rag.config_rag import RAGConfig


class DBIndexer:
    """
    Offline-only.
    Builds (or rebuilds) a persisted Chroma vector DB from docx files.
    """

    def __init__(self, cfg: RAGConfig):
        self.cfg = cfg

    def load_documents(self) -> List[Document]:
        """Load all .docx files from the configured docs directory"""
        docs: List[Document] = []

        # Ensure docs directory exists
        if not self.cfg.docs_dir.exists():
            raise FileNotFoundError(
                f"Documents directory not found: {self.cfg.docs_dir}\n"
                f"Please create it or check your project_root setting."
            )

        # Load all .docx files
        docx_files = list(self.cfg.docs_dir.rglob("*.docx"))
        if not docx_files:
            raise ValueError(
                f"No .docx files found in {self.cfg.docs_dir}\n"
                f"Please add documents to index."
            )

        print(f"Found {len(docx_files)} .docx file(s) in {self.cfg.docs_dir}")

        for path in docx_files:
            loader = Docx2txtLoader(str(path))
            docs.extend(loader.load())

        return docs

    def split_documents(self, docs: List[Document]) -> List[Document]:
        """Split documents into chunks using configured parameters"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.cfg.chunk_size,
            chunk_overlap=self.cfg.chunk_overlap,
            separators=list(self.cfg.separators),
        )
        return splitter.split_documents(docs)

    def build(self, wipe: bool = False) -> Chroma:
        """
        Build the vector database.

        Args:
            wipe: If True, delete existing database before building

        Returns:
            The created Chroma vector database
        """
        # Wipe existing database if requested
        if wipe and self.cfg.persist_dir.exists():
            print(f"Wiping existing database at {self.cfg.persist_dir}")
            shutil.rmtree(self.cfg.persist_dir, ignore_errors=True)

        # Load and split documents
        print("Loading documents...")
        docs = self.load_documents()
        print(f"Loaded {len(docs)} document(s)")

        print("Splitting into chunks...")
        chunks = self.split_documents(docs)
        print(f"Created {len(chunks)} chunk(s)")

        # Create embeddings
        print(f"Creating embeddings using {self.cfg.embedding_model}...")
        embedding = OpenAIEmbeddings(model=self.cfg.embedding_model)

        # Build vector database
        print(f"Building Chroma database at {self.cfg.persist_dir}...")
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            persist_directory=str(self.cfg.persist_dir),
            collection_name=self.cfg.collection_name,
        )

        # Sanity check
        count = vector_db._collection.count()
        if count != len(chunks):
            raise RuntimeError(
                f"Chroma count mismatch: chunks={len(chunks)} db={count}"
            )

        print(f"✓ Successfully built database with {count} chunks")
        return vector_db