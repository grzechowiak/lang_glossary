from __future__ import annotations

from typing import List
import shutil
from pathlib import Path
import pandas as pd

from langchain_core.documents import Document
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from rag.config_rag import RAGConfig
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M")


class DBIndexer:
    """
    Offline-only.
    Builds (or rebuilds) a persisted Chroma vector DB from docx files.
    """

    def __init__(self, cfg: RAGConfig):
        self.cfg = cfg

    @staticmethod
    def _load_docx(path: Path) -> List[Document]:
        loaded = Docx2txtLoader(str(path)).load()
        for d in loaded:
            d.metadata.update({"source": str(path), "file_type": "docx"})
        return loaded

    @staticmethod
    def _load_csv(path: Path) -> List[Document]:
        df = pd.read_csv(path, delimiter=';')
        return [
            Document(
                page_content=df.to_csv(index=False),
                metadata={
                    "source": str(path),
                    "file_type": "csv",
                },
            )
        ]

    def load_documents(self) -> List[Document]:
        docs: List[Document] = []

        sources = {
            self.cfg.docs_dir: {".docx": self._load_docx},
            self.cfg.excel_dir: {".csv": self._load_csv},  # using same folder for csv
        }

        for folder in sources:
            if not folder.exists():
                raise FileNotFoundError(f"Source directory not found: {folder}")

        file_count = 0
        for folder, handlers in sources.items():
            for ext, loader_fn in handlers.items():
                files = list(folder.rglob(f"*{ext}"))
                file_count += len(files)
                print(f"Found {len(files)} {ext} file(s) in {folder}")
                for path in files:
                    docs.extend(loader_fn(path))

        if file_count == 0:
            raise ValueError(
                f"No supported files found in:\n"
                f"- {self.cfg.docs_dir}\n"
                f"- {self.cfg.excel_dir}"
            )

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
        by_type = {}
        for d in docs:
            t = d.metadata.get("file_type", "unknown")
            by_type[t] = by_type.get(t, 0) + 1
        print(f"Loaded {len(docs)} document(s): {by_type}")

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
            collection_metadata={"created_at": timestamp},
        )

        # Sanity check
        count = vector_db._collection.count()
        if count != len(chunks):
            raise RuntimeError(
                f"Chroma count mismatch: chunks={len(chunks)} db={count}"
            )

        print(f"✓ Successfully built database with {count} chunks")
        return vector_db