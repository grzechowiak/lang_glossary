from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RAGConfig:
    """
    RAG Configuration with all paths relative to project_root.
    Only project_root needs to be set - all other paths are derived from it.
    """

    # The single source of truth for all paths
    project_root: Path

    # Relative path configurations
    docs_dirname: str = "data/docs"
    persist_dirname: str = "vector_dbs/chroma_db"

    # Model and chunking parameters
    collection_name: str = "business_glossary"
    embedding_model: str = "text-embedding-3-small"
    chunk_size: int = 300
    chunk_overlap: int = 50
    chunk_retrieve_default: int = 1
    separators: tuple = ("\n\n", "\n", " ", "")

    @property
    def docs_dir(self) -> Path:
        """Directory containing source .docx files"""
        return self.project_root / self.docs_dirname

    @property
    def persist_dir(self) -> Path:
        """Directory where Chroma DB will be persisted"""
        return self.project_root / self.persist_dirname

    def __post_init__(self):
        """Ensure project_root is a Path object"""
        if not isinstance(self.project_root, Path):
            object.__setattr__(self, 'project_root', Path(self.project_root))
