# component_2/llm.py
import json
from typing import Dict, List

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from component_two.configs.config import ConfigAgent

from dotenv import load_dotenv
load_dotenv(override=True)

# Instantiate configuration
confing_constants = ConfigAgent()

# Initialize the LLM client
_llm = ChatOpenAI(model=confing_constants.llm_model, temperature=confing_constants.llm_temperature)


def regenerate_row_with_llm(columns: List[str], current_row: Dict[str, object], feedback: str) -> Dict[str, object]:
    """
    Use a language model to regenerate a row based on user feedback.

    The function constructs a prompt that includes the current row data and user feedback,
    then uses the LLM to generate a new version of the row that incorporates the feedback
    while following specific formatting rules.

    Args:
        columns: List of column names that should be included in the result
        current_row: Dictionary containing the current row data
        feedback: User's natural language feedback on how to modify the row

    Returns:
        Dictionary containing the regenerated row data with the same columns

    Raises:
        ValueError: If the LLM returns invalid JSON or non-dictionary data
    """
    # Create system message with instructions
    system = SystemMessage(
        content=(
            "You rewrite ONE CSV row.\n"
            "Return ONLY valid JSON.\n"
            "The JSON MUST be an object with keys EXACTLY equal to the provided Columns.\n"
            "No extra keys. No markdown. No commentary."
        )
    )

    # Create human message with current data and feedback
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

    # Get response from LLM
    raw = _llm.invoke([system, human]).content.strip()

    # Parse and validate response
    try:
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            raise ValueError("LLM JSON is not an object")
        return obj
    except Exception as exc:
        raise ValueError(f"LLM returned invalid JSON: {raw!r}") from exc