import pandas as pd

class ConfigDatasets:
    """Configuration class"""

    def __init__(self):
        # Column mappings for CSV file (Business Glossary Master)
        self.column_mappings_master_bg = {
            "Bucket": "bucket_name",
            "Dataset": "dataset_name",
            "Table Name": "table_name",
            "Column Name": "column_name",
            "Business Domain Name": "business_domain_name",
            "Business Sub-Domain Name": "business_sub_domain_name",
            "Business Name": "business_name",
            "Column Description": "column_description",
            "Sample Values": "sample_values",
            "Attribute related business rationale": "attribute_rationale",
            "Attribute logical business rules": "attribute_rule",
            "Data Owner Name": "data_owner_name",
            "Data Owner E-Mail": "data_owner_email",
            # "Data Steward Approval": "data_steward_approval",
            # "Data Steward Feedback (only if not approved)": "data_steward_feedback",
            # "Data Owner Approval": "data_owner_approval",
            # "Data Owner Feedback (only if not approved)": "data_owner_feedback",
        }

        # Column mappings for CSV file (Data Owners Master)
        self.column_mappings_master_data_owners = {
            "Bucket": 'bucket_name',
            "Dataset": "dataset_name",
            "Table Name": "table_name",
            "Data Steward Name": "steward_name",
            "Data Steward E-Mail": "steward_email",
            "Data Owner Name": "data_owner_name",
            "Data Owner E-Mail": "data_owner_email"
        }

        # Columns needed to define the template
        self.define_columns_to_fill = {
            'key_columns': [
                'bucket_name',
                'dataset_name',
                'table_name',
                'column_name'
            ],
            'rag_columns': [
                "business_domain_name",
                "business_sub_domain_name",
                "business_name",
                "column_description",
                "attribute_rationale",
                "attribute_rule"
            ],
            'data_steward_columns': [
                "data_owner_name",
                "data_owner_email"
            ],
            'additional_columns': [
                "sample_values"
            ],
            # 'columns_to_drop': [
            #     "data_steward_approval",
            #     "data_steward_feedback",
            #     "data_owner_approval",
            #     "data_owner_feedback"
            # ]
        }

        # Values which will be taken from the GCP system
        self.bucket_name_value = 'gs_bucket'
        self.dataset_name_value = 'client'
        self.table_name_value = "client_account"

        # Placeholders
        self.rag_placeholder = "<agent>"
        self.ds_placeholder = "<ds_master>"

        # self.rag_columns = [
        #     "business_domain_name",
        #     "business_sub_domain_name",
        #     "business_name",
        #     "column_description",
        #     "attribute_rationale",
        #     "attribute_rule"
        # ]
        #
        # self.data_steward_columns = [
        #     "data_owner_name",
        #     "data_owner_email"
        # ]
        #
        # self.additional_columns = [
        #     "sample_values"
        # ]
        #
        # self.columns_to_drop = [
        #     "data_steward_approval",
        #     "data_steward_feedback",
        #     "data_owner_approval",
        #     "data_owner_feedback"
        # ]

        #
        #
        # # Agent settings
        # self.max_iterations = 3
        # self.recursion_limit = 20
        # self.llm_model = 'gpt-4.1-nano' #"gpt-4o" #"gpt-4o-mini" #'gpt-4.1-nano'
        # self.llm_temperature = 0.0

    def get_framework_dict(self):
        """Return framework settings in the format expected by nodes."""
        return {
            "key_col": list(self.define_columns_to_fill["key_columns"]),
            "additional_col": list(self.define_columns_to_fill["additional_columns"]),
            "search_with_RAG": list(self.define_columns_to_fill["rag_columns"]),
            "search_with_data_steward_file": list(self.define_columns_to_fill["data_steward_columns"]),
        }

    def template_columns(self):
        fw = self.get_framework_dict()
        cols = (
                fw["key_col"]
                + fw["additional_col"]
                + fw["search_with_RAG"]
                + fw["search_with_data_steward_file"]
        )
        # de-dupe preserve order
        return list(dict.fromkeys(cols))


    def build_template(self, original_sample_dict, bucket=None, dataset=None, table=None):
        df0 = pd.DataFrame(original_sample_dict)

        df = pd.DataFrame({
            "column_name": df0.columns,
            "sample_values": [df0[c].tolist() for c in df0.columns],
        })

        df["bucket_name"] = bucket or self.bucket_name_value
        df["dataset_name"] = dataset or self.dataset_name_value
        df["table_name"] = table or self.table_name_value

        for c in self.define_columns_to_fill["rag_columns"]:
            df[c] = self.rag_placeholder

        for c in self.define_columns_to_fill["data_steward_columns"]:
            df[c] = self.ds_placeholder

        # enforce schema/order from config
        cols = self.template_columns()
        for c in cols:
            if c not in df.columns:
                df[c] = None
        return df[cols]

    # @property
    # def columns_to_drop(self) -> list[str]:
    #     """
    #     Map canonical column names to original CSV headers
    #     using column_mappings_master_bg
    #     """
    #     canonical_to_drop = self.define_columns_to_fill.get("columns_to_drop", [])
    #
    #     return [
    #         original_name
    #         for original_name, canonical_name in self.column_mappings_master_bg.items()
    #         if canonical_name in canonical_to_drop
    #     ]
