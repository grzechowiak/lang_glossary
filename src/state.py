import operator
from typing import List, TypedDict, Annotated, Dict, Any
from pydantic import BaseModel, Field


# --- Pydantic Models for Output Structure ---

class ColumnDefInput(BaseModel):
    bucket_name: str = Field(
        description="The name of the bucket name to which dataset belong.")
    dataset_name: str = Field(
        description="The name of the dataset to which the table belong..")
    table_name: str = Field(
        description="The logical or physical name of the database table to which the columns belong.")
    column_name: str = Field(
        description="The exact (physical) name of the column from the source.")

    sample_values: List[str] = Field(default=[],
                                     description="Representative example values illustrating valid content and typical usage.")
    business_domain_name: str = Field(
        description="The high-level business domain representing a major business capability or functional area.")
    business_sub_domain_name: str = Field(
        description="A more granular grouping based on specific business activities. The field 'business_sub_domain_name' must be a more granular compared to the 'business_domain_name")
    business_name: str = Field(
        description="The logical, business-friendly semantic name of the attribute. Convert technical column names (e.g., `CUST_ID_01`) into logical, Title Case names (e.g., `Customer Identifier`). Expand all abbreviations.")
    column_description: str = Field(
        description="A clear and concise definition of what the data element represents in business terms. Provide a functional definition. Focus on business value; do not refer to technical data types.")
    attribute_rationale: str = Field(
        description="Explanation of why the attribute is required and how it supports processes or regulatory needs.")
    attribute_rule: str = Field(
        description="The logical constraints, validations, or conditions that govern usage.")

    data_owner_name: str = Field(
        description="The role or individual responsible for the data element's quality and governance.")
    data_owner_email: str = Field(
        description="Contact information for the data owner overseeing this attribute.")

class ColumnDefOutput(ColumnDefInput):
    extra__add_citation_of_the_hit: str = Field(
        description="A citation of the hit where the information was found. Three possible values can be placed here: 1) If using RAG, extract the verbatim sentence. 2) If the fields was already filled, say: 'Master Business Glossary'. 3) If none of the previous and Agent invented it with no source, say 'Agent Logic'")
    extra__add_source_explained: str = Field(
        description="The exact value of source where the definition was created. Three possible values can be placed here: 1) If using RAG, provide the exact 'source' value. 2) If the fields was already filled, say: 'Master Business Glossary'. 3) If none of the previous and Agent invented it with no source, say 'Agent Logic'")


class TemplateOutput(BaseModel):
    rows: List[ColumnDefOutput]
    table_summary: str = Field("Table Level summary of the entire table which would provide a high-level information what this table is about, what kind of information it contains, and what is the main purpose of this table. This summary should be concise, ideally not more than 2-3 sentences.")


class ValidationResult(BaseModel):
    is_valid: bool = Field(
        description="True if all definitions are sensible and all columns are present.")
    feedback: str = Field(
        description="If is_valid is False, provide specific instructions on what to fix.")


# --- LangGraph State Definition ---

class AgentState(TypedDict, total=False):
    # Configuration & Inputs
    framework_def: Dict[str, Any]
    source_original_table: Dict[str, List[Any]]
    master_business_glossary: Dict[str, List[Any]]
    master_data_owner: Dict[str, List[Any]]

    # Intermediate RAG Data
    RAG_cols_with_samples: Dict[str, List[Any]]
    RAG_company_context: str

    # Working Context
    entire_table_context: Dict[str, List[Any]]
    template_df: Dict[str, List[Any]]

    # Outputs & Control Flow
    result: TemplateOutput
    error_message: str
    iterations: int
    review_history_validator: Annotated[List[str], operator.add]