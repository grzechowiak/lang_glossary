import io
from typing import Any, Dict, List
from datetime import datetime

import pandas as pd
from fastapi import HTTPException

from component_one.src.state import ColumnDefOutput
from component_one.configs.config_paths import ConfigPaths


# Configuration class
class ColumnConfig:
    """Configuration related to column definitions and CSV format."""

    # Get configuration values
    config_paths = ConfigPaths()

    # The canonical column order from Pydantic model fields
    CANONICAL_COLUMNS: List[str] = list(ColumnDefOutput.model_fields.keys())

    # CSV separator from configuration
    CSV_SEPARATOR = config_paths.csv_separator

    # Output directory
    OUTPUT_DIR = config_paths.output_dir_component_two_result

    # File name to save
    OUTPUT_FILENAME = config_paths.output_filename_revised_table


class DataUtils:
    """Utility functions for data manipulation, validation, and persistence."""

    @staticmethod
    def coerce_sample_values(value: Any) -> str:
        """
        Normalize sample values to a string.

        Since the input is expected to be a string already (from the Pydantic model),
        this function mainly handles None/empty values and does basic string cleaning.
        """
        # Handle None or empty cases
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return ""
        # For strings, just clean it up
        if isinstance(value, str):
            return value.strip()
        # If missing a case, then convert to str
        return str(value)

    @classmethod
    def validate_dataframe(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate a DataFrame against the expected schema using Pydantic models.

        Args:
            df: DataFrame to validate

        Returns:
            Validated DataFrame with normalized values

        Raises:
            HTTPException: If validation fails
        """
        # Ensure all expected columns exist
        missing = [c for c in ColumnConfig.CANONICAL_COLUMNS if c not in df.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"DataFrame is missing columns: {missing}")

        # Keep only canonical columns and in canonical order
        df = df[ColumnConfig.CANONICAL_COLUMNS].copy()

        # Normalize NaN to None for validation
        records: List[Dict[str, Any]] = df.where(pd.notnull(df), None).to_dict(orient="records")

        validated_rows: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []

        for idx, row in enumerate(records):
            # normalize sample_values
            row["sample_values"] = cls.coerce_sample_values(row.get("sample_values"))

            try:
                model = ColumnDefOutput(**row)
                validated_rows.append(model.model_dump())
            except Exception as e:
                errors.append({"row_index": idx, "error": str(e)})

        if errors:
            # return first few errors for readability
            raise HTTPException(status_code=400, detail={"message": "Validation failed", "errors": errors[:10]})

        return pd.DataFrame(validated_rows, columns=ColumnConfig.CANONICAL_COLUMNS)

    @classmethod
    def load_csv_and_validate(cls, raw: bytes) -> pd.DataFrame:
        """
        Parse CSV bytes, validate rows via Pydantic, and return DataFrame.

        Raises HTTPException if validation fails.
        """
        try:
            df = pd.read_csv(io.BytesIO(raw), sep=ColumnConfig.CSV_SEPARATOR)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid CSV: {e}")

        # Use the separate validation function
        return cls.validate_dataframe(df)

    @staticmethod
    def save_dataframe_to_csv(df: pd.DataFrame, filename_base: str) -> str:
        """
        Save a DataFrame to a CSV file with configurable options.

        Args:
            df: The pandas DataFrame to save
            filename_base: Base name for the output file

        Returns:
            Path to the saved file as string
        """
        output_dir = ColumnConfig.OUTPUT_DIR
        csv_separator = ColumnConfig.CSV_SEPARATOR
        output_filename = ColumnConfig.OUTPUT_FILENAME

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Full path for output file
        output_path = output_dir / f"{output_filename}.csv"

        # Save DataFrame to CSV using the configured separator
        df.to_csv(
            output_path,
            sep=csv_separator,
            index=False
        )

        print(f"Saved CSV file to: {output_path}")
        return str(output_path)