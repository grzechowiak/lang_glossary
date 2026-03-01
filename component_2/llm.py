import json
from typing import Dict, List

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from dotenv import load_dotenv
load_dotenv(override=True)

LLM_MODEL_NAME = "gpt-4.1-mini"
LLM_TEMPERATURE = 0.0

_llm = ChatOpenAI(model=LLM_MODEL_NAME, temperature=LLM_TEMPERATURE)


def regenerate_row_with_llm(columns: List[str], current_row: Dict[str, object], feedback: str) -> Dict[str, object]:
    system = SystemMessage(
        content=(
            "You rewrite ONE CSV row.\n"
            "Return ONLY valid JSON.\n"
            "The JSON MUST be an object with keys EXACTLY equal to the provided Columns.\n"
            "No extra keys. No markdown. No commentary."
        )
    )

    human = HumanMessage(
        content=(
            f"Columns: {json.dumps(columns)}\n"
            f"Current row (JSON): {json.dumps(current_row, ensure_ascii=False)}\n\n"
            f"Human feedback (free text):\n{feedback}\n\n"
            "Rules:\n"
            "- Keep bucket_name, dataset_name, table_name, column_name unchanged unless explicitly asked.\n"
            "- sample_values MUST be a comma-separated string (e.g. 'value1, value2, value3').\n"
            "- Return ONLY the JSON object."
        )
    )

    raw = _llm.invoke([system, human]).content.strip()

    try:
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            raise ValueError("LLM JSON is not an object")
        return obj
    except Exception as exc:
        raise ValueError(f"LLM returned invalid JSON: {raw!r}") from exc