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
        self.output_dir = self.project_root / "99_playground"

        # CSV file paths
        self.main_dataset = self.data_dir / "datasets" / "dataset_csv.csv"
        self.master_glossary = self.data_dir / "master_business_glossary" / "master_business_glossary_csv.csv"
        self.data_stewards = self.data_dir / "stewards_and_owners" / "data_stewards.csv"

        # Output settings
        self.output_filename_prefix = "BG_TEST_v1"
        self.include_timestamp = True

        # CSV settings
        self.csv_separator = ";"

        # # Framework settings
        # self.bucket_name = 'gs_bucket'
        # self.dataset_name = 'client'
        # self.table_name = "client_account"


    #     # Column mappings for CSV files
    #     self.column_mappings = {
    #         # External CSV names : Internal names used
    #         "Bucket": "bucket_name",
    #         "Dataset": "dataset_name",
    #         "Table Name": "table_name",
    #         "Column Name": "column_name",
    #         "Business Domain Name": "business_domain_name",
    #         "Business Sub-Domain Name": "business_sub_domain_name",
    #         "Business Name": "business_name",
    #         "Column Description": "column_description",
    #         "Sample Values": "sample_values",
    #         "Attribute related business rationale": "attribute_rationale",
    #         "Attribute logical business rules": "attribute_rule",
    #         "Data Owner Name": "data_owner_name",
    #         "Data Owner E-Mail": "data_owner_email",
    #         "Data Steward Approval": "data_steward_app",
    #         "Data Steward Feedback (only if not approved)": "data_steward_feedback",
    #         "Data Owner Approval": "data_owner_approval",
    #         "Data Owner Feedback (only if not approved)": "data_owner_feedback",
    #     }
    #
    #
    #     self.rag_columns = [
    #         "Business Domain Name",
    #         "Business Sub-Domain Name",
    #         "Business Name",
    #         "Column Description",
    #         "Attribute related business rationale",
    #         "Attribute logical business rules"
    #     ]
    #
    #     self.data_steward_columns = [
    #         "Data Owner Name",
    #         "Data Owner E-Mail"
    #     ]
    #
    #     self.additional_columns = ["Sample Values"]
    #
    #     self.columns_to_drop = [
    #         "Data Steward Approval",
    #         "Data Steward Feedback (only if not approved)",
    #         "Data Owner Approval",
    #         "Data Owner Feedback (only if not approved)"
    #     ]
    #
    #     # Output settings
    #     self.output_filename_prefix = "BG_TEST_v1"
    #     self.include_timestamp = True
    #
    #     # Agent settings
    #     self.max_iterations = 3
    #     self.recursion_limit = 20
    #     self.llm_model = 'gpt-4.1-nano' #"gpt-4o" #"gpt-4o-mini" #'gpt-4.1-nano'
    #     self.llm_temperature = 0.0
    #
    # def get_framework_dict(self):
    #     """Return framework settings in the format expected by nodes."""
    #     return {
    #         "bucket_name": self.bucket_name,
    #         "dataset_name": self.dataset_name,
    #         "table_name_value": self.table_name,
    #         "additional_col": self.additional_columns,
    #         "search_with_RAG": self.rag_columns,
    #         "search_with_data_steward_file": self.data_steward_columns,
    #     }
