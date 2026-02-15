import operator
from typing import List, TypedDict, Annotated, Dict, Any
from pydantic import BaseModel, Field


# --- Pydantic Models for Output Structure ---

class ColumnDef(BaseModel):
    table_name: str = Field(
        description="The logical or physical name of the database table to which the attribute belongs.")
    column_name: str = Field(
        description="The exact (physical) name of the column from the source.")
    sample_values: List[str] = Field(default=[],
                                     description="Representative example values illustrating valid content and typical usage.")
    business_domain_name: str = Field(
        description="The high-level business domain representing a major business capability or functional area.")
    business_sub_domain_name: str = Field(
        description="A more granular grouping based on specific business activities.")
    business_name: str = Field(
        description="The logical, business-friendly semantic name of the attribute.")
    column_description: str = Field(
        description="A clear and concise definition of what the data element represents in business terms.")
    business_rationale: str = Field(
        description="Explanation of why the attribute is required and how it supports processes or regulatory needs.")
    logical_business_rules: str = Field(
        description="The logical constraints, validations, or conditions that govern usage.")
    data_owner: str = Field(
        description="The role or individual responsible for the data element's quality and governance.")
    data_owner_email: str = Field(
        description="Contact information for the data owner overseeing this attribute.")
    extra__add_citation_of_the_hit: str = Field(
        description="A citation of the hit where the information was found.")
    extra__add_source_explained: str = Field(
        description="The exact value of source where the definition was created.")


class TemplateOutput(BaseModel):
    columns: List[ColumnDef]


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
    result: List[ColumnDef]
    error_message: str
    iterations: int
    review_history_validator: Annotated[List[str], operator.add]