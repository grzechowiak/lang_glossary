from pathlib import Path
from datetime import datetime
from component_one.configs.config_gcp_info import ConfigFetchBucketDatasetTable

# Get config for saving specific table
cfg_gcp_info = ConfigFetchBucketDatasetTable()
bucket_name = cfg_gcp_info.bucket_name_value
dataset_name = cfg_gcp_info.dataset_name_value
table_name = cfg_gcp_info.table_name_value

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

class ConfigPaths:
    """Configuration class"""

    def __init__(self):
        """
        Initialize all configuration paths.
        """

        # Project paths
        this_file = Path(__file__).resolve()
        configs_dir_level2 = this_file.parent
        component_dir_level1 = configs_dir_level2.parent
        project_root_level0 = component_dir_level1.parent

        print("")
        print("-- DIR Paths --")
        print('configs_dir:', configs_dir_level2)
        print('DIR LVL. 1: component1_dir:', component_dir_level1)
        print('DIR LVL. 0: project_root:', project_root_level0)

        self.configs_dir_level2 = configs_dir_level2
        self.component_dir_level1 = component_dir_level1
        self.project_root_level0 = project_root_level0

        # Define main DIRs
        self.data_dir = self.component_dir_level1 / "data"
        self.output_dir = self.project_root_level0 / "outputs_savings"
        # Component one DIR
        self.output_dir_specific_table = self.output_dir / f"{bucket_name}_{dataset_name}_{table_name}" / "component_one" / timestamp
        # Component two DIR
        self.output_dir_component_two_result = self.output_dir / f"{bucket_name}_{dataset_name}_{table_name}" / "component_two" / timestamp

        # CSV file paths
        self.main_dataset = self.data_dir / "datasets" / "dataset_csv.csv"
        self.master_glossary = self.data_dir / "master_business_glossary" / "master_business_glossary_csv.csv"
        self.data_stewards = self.data_dir / "stewards_and_owners" / "data_stewards.csv"

        # Output file names for Component 1
        self.output_filename_final_table = "FINAL_TABLE"
        self.output_filename_context_rag = "CONTEXT"
        self.output_filename_table_summary = "TABLE_SUMMARY"

        # Output filenames for Component 2
        self.output_filename_revised_table = "FINAL_TABLE_VERIFIED"


        # CSV settings
        self.csv_separator = ";"