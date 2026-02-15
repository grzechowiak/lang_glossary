from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from rag.config_rag import RAGConfig
from rag.vector_retriever import VectorRetriever

import pandas as pd
import json


@dataclass
class RetrievedResults:
    column_name: str
    query: str
    hits: List[Dict[str, Any]]  # Each hit has: {"page_content": str, "metadata": dict}


class PrepareRetrieval:
    """
    Set of tools which given a dataframe, can:
        - sample values from each column,
        - retrieve relevant info for each column from vector DB,
        - and format it into a text prompt.
    """

    def __init__(self, cfg: RAGConfig):
        self.cfg = cfg
        self.retriever = VectorRetriever(cfg)
        self.retriever.load_vector_db()


    ## 1st for each column call retriever (using retrieve_for_single_column)
    def retrieve_for_all_columns(self, col_samples: Dict[str, List[str]]) -> Dict[str, RetrievedResults]:
        results: Dict[str, RetrievedResults] = {}

        for col, samples in col_samples.items():
            results[col] = self.retrieve_for_single_column(column_name=col, sample_values=samples)

        print(f"\n✅ Retrieved all relevant contextual data chunks for {len(col_samples)} columns.")
        return results

    def retrieve_for_single_column(self, column_name: str, sample_values: Any, ) -> RetrievedResults:
        if sample_values is None:
            sample_values = []

        # Transform values into strings
        sample_values = [str(val) for val in sample_values]
        examples = ", ".join([str(v) for v in sample_values if v][:5])

        query = (
            f'Find relevant information for the following variable: '
            f"Column Name: '{column_name}'. "
            f"Example Values: {examples}. "
        )

        docs = self.retriever.retrieve(query=query)

        hits: List[Dict[str, Any]] = []
        for d in docs:
            hits.append({
                    "page_content": d.page_content,  # The text
                    "metadata": d.metadata,          # Dict with 'source', etc.
             })

        return RetrievedResults(
            column_name=column_name,
            query=query,
            hits=hits,
        )

    ## 2nd: build context prompt based on retrieved info
    @staticmethod
    def build_prompt_and_format(results: Dict[str, RetrievedResults]) -> str:
        payload = {"columns": []}

        for col_key, res in results.items():
            column_block = {
                "column_name": res.column_name,
                "hits": []
            }

            for idx, hit in enumerate(res.hits or [], start=1):
                text = hit.get("page_content", "").replace("\n", " ").strip()
                source = hit.get("metadata", {}).get("source", "unknown")

                column_block["hits"].append({
                    "hit_id": f"{col_key}#{idx}",
                    "text": text,
                    "source": source
                })

            payload["columns"].append(column_block)

        print(f"Formatted context for {len(payload['columns'])} columns is ready for the Agent.")

        return json.dumps(payload, indent=2, ensure_ascii=False)
