from pathlib import Path

class ConfigPaths:
    """Configuration class"""

    def __init__(self, project_root: Path = None):
        """
        Initialize configuration.

        Args:
            project_root: Base directory for the project. Defaults to current directory.
        """
        # Project paths
        self.include_timestamp = None
        self.project_root = project_root or Path.cwd()
        self.data_dir = self.project_root / "data"
        self.output_dir = self.project_root / "99_playground" / "outputs_collected"

        # CSV file paths
        self.main_dataset = self.data_dir / "datasets" / "dataset_csv.csv"
        self.master_glossary = self.data_dir / "master_business_glossary" / "master_business_glossary_csv.csv"
        self.data_stewards = self.data_dir / "stewards_and_owners" / "data_stewards.csv"

        # Output settings
        self.output_filename_suffix_context_rag = "BG_CONTEXT"
        self.output_filename_suffix_table_summary = "BG_TABLE_SUMMARY"
        self.output_filename_suffix_final_table = "BG_FINAL_TABLE"
        self.include_timestamp = True

        # CSV settings
        self.csv_separator = ";"