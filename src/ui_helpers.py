import gradio as gr
import pandas as pd
import json
import os
import tempfile
import time


def display_with_gradio(columns):
    """
    Create a simple Gradio interface for reviewing the business glossary.
    """
    # Create a temporary file to store the result
    temp_dir = tempfile.gettempdir()
    result_file = os.path.join(temp_dir, "gradio_result.txt")

    # Delete any existing result file
    if os.path.exists(result_file):
        os.remove(result_file)

    # Convert columns to a DataFrame for display
    data = []
    for i, col in enumerate(columns):
        col_dict = col.dict()
        col_dict['index'] = i + 1
        data.append(col_dict)

    df = pd.DataFrame(data)

    # Select relevant columns for display
    display_columns = ['index', 'column_name', 'business_name', 'column_description',
                       'business_domain_name', 'business_sub_domain_name',
                       'business_rationale']

    # Rename columns for better display
    column_names = {
        'index': '#',
        'column_name': 'Column Name',
        'business_name': 'Business Name',
        'column_description': 'Description',
        'business_domain_name': 'Business Domain',
        'business_sub_domain_name': 'Business Sub-Domain',
        'business_rationale': 'Business Rationale'
    }

    # Use only columns that exist in the dataframe
    available_columns = [col for col in display_columns if col in df.columns]
    display_df = df[available_columns].rename(columns={col: column_names.get(col, col) for col in available_columns})

    # Track modifications
    current_columns = [col.copy() for col in columns]  # Deep copy
    modifications_made = []

    # Helper function to save result
    def save_result(result_data):
        with open(result_file, "w") as f:
            f.write(json.dumps(result_data))
        return "✅ Result saved. You can close this interface now."

    # Action handlers
    def approve_all():
        result = {
            "action": "approve",
            "data": {
                "human_approved": True,
                "human_feedback": "Approved by human reviewer",
                "human_review_history": ["Human reviewer approved all entries"]
            }
        }

        # If modifications were made, include them
        if modifications_made:
            result["data"]["result"] = [col.dict() for col in current_columns]
            result["data"]["human_feedback"] = f"Modified {len(modifications_made)} entries and approved"
            result["data"]["human_review_history"] = modifications_made

        return save_result(result)

    def request_regeneration(feedback):
        if not feedback.strip():
            return "⚠️ Please provide feedback for regeneration."

        result = {
            "action": "regenerate",
            "data": {
                "human_approved": False,
                "human_feedback": feedback,
                "human_review_history": [f"Human reviewer requested regeneration: {feedback}"]
            }
        }
        return save_result(result)

    # Function to get entry details for modification
    def get_entry_details(entry_index):
        try:
            idx = int(entry_index) - 1  # Convert to 0-based index
            if 0 <= idx < len(current_columns):
                col = current_columns[idx]
                return (
                    gr.update(value=col.business_name),
                    gr.update(value=col.column_description),
                    gr.update(value=col.business_domain_name),
                    gr.update(value=col.business_sub_domain_name),
                    gr.update(value=col.business_rationale),
                    gr.update(visible=True)
                )
            else:
                return (
                    gr.update(value=""),
                    gr.update(value=""),
                    gr.update(value=""),
                    gr.update(value=""),
                    gr.update(value=""),
                    gr.update(visible=False)
                )
        except ValueError:
            return (
                gr.update(value=""),
                gr.update(value=""),
                gr.update(value=""),
                gr.update(value=""),
                gr.update(value=""),
                gr.update(visible=False)
            )

    # Function to update entry
    def submit_modifications(entry_idx, business_name, description, domain, subdomain, rationale):
        try:
            idx = int(entry_idx) - 1
            if 0 <= idx < len(current_columns):
                # Track changes
                changes = []

                # Apply changes
                if business_name.strip():
                    current_columns[idx].business_name = business_name
                    changes.append("business name")

                if description.strip():
                    current_columns[idx].column_description = description
                    changes.append("description")

                if domain.strip():
                    current_columns[idx].business_domain_name = domain
                    changes.append("business domain")

                if subdomain.strip():
                    current_columns[idx].business_sub_domain_name = subdomain
                    changes.append("business sub-domain")

                if rationale.strip():
                    current_columns[idx].business_rationale = rationale
                    changes.append("business rationale")

                # Update metadata
                if changes:
                    current_columns[idx].extra__add_source_explained = "Human Review"
                    current_columns[idx].extra__add_citation_of_the_hit = "Modified by human reviewer"

                    # Record modification
                    modifications_made.append(
                        f"Modified {', '.join(changes)} for column {current_columns[idx].column_name}")

                    # Update display data
                    data = []
                    for i, col in enumerate(current_columns):
                        col_dict = col.dict()
                        col_dict['index'] = i + 1
                        data.append(col_dict)

                    new_df = pd.DataFrame(data)
                    display_df = new_df[available_columns].rename(
                        columns={col: column_names.get(col, col) for col in available_columns})

                    return f"✅ Changes saved for entry {entry_idx}. You can modify more entries or approve when done.", display_df
                else:
                    return "No changes made. Please enter values to modify.", None
            else:
                return f"Invalid entry index: {entry_idx}", None
        except ValueError:
            return f"Invalid entry index format", None

    # Create Gradio interface
    with gr.Blocks(title="Business Glossary Review") as demo:
        gr.Markdown("# Business Glossary Review")
        gr.Markdown("Review the generated business glossary and choose an action:")

        # Display table
        data_display = gr.DataFrame(value=display_df)

        # Action tabs
        with gr.Tabs():
            with gr.TabItem("✅ Approve & Finish"):
                approve_btn = gr.Button("Approve All Entries")
                approve_output = gr.Textbox(label="Status")
                approve_btn.click(approve_all, inputs=[], outputs=approve_output)

            with gr.TabItem("🔄 Request Regeneration"):
                regen_feedback = gr.Textbox(label="Feedback for Regeneration",
                                            placeholder="Please describe what needs to be improved...",
                                            lines=3)
                regen_btn = gr.Button("Request Regeneration")
                regen_output = gr.Textbox(label="Status")
                regen_btn.click(request_regeneration, inputs=[regen_feedback], outputs=regen_output)

            with gr.TabItem("✏️ Modify Entries"):
                # Selection
                entry_selector = gr.Textbox(label="Entry Number to Modify", placeholder="Enter a number (e.g., 2)")
                load_btn = gr.Button("Load Entry")

                # Modification fields
                with gr.Column(visible=False) as mod_fields:
                    business_name_input = gr.Textbox(label="Business Name")
                    description_input = gr.Textbox(label="Description", lines=3)
                    domain_input = gr.Textbox(label="Business Domain")
                    subdomain_input = gr.Textbox(label="Business Sub-domain")
                    rationale_input = gr.Textbox(label="Business Rationale", lines=2)

                    submit_btn = gr.Button("Submit Changes")
                    mod_output = gr.Textbox(label="Status")

                # Hook up events
                load_btn.click(
                    get_entry_details,
                    inputs=[entry_selector],
                    outputs=[business_name_input, description_input, domain_input,
                             subdomain_input, rationale_input, mod_fields]
                )

                submit_btn.click(
                    submit_modifications,
                    inputs=[entry_selector, business_name_input, description_input,
                            domain_input, subdomain_input, rationale_input],
                    outputs=[mod_output, data_display]
                )

                # Instructions
                gr.Markdown("""
                ### Instructions:
                1. Enter the entry number and click "Load Entry"
                2. Modify the fields as needed
                3. Click "Submit Changes" to apply
                4. Repeat for other entries
                5. When finished, go to "Approve & Finish" tab
                """)

    # Launch Gradio
    print("\n✅ Opening Gradio interface for review...")
    demo.launch(share=False, inbrowser=True)

    # Wait for result
    print("Waiting for your feedback...")
    while not os.path.exists(result_file):
        time.sleep(0.5)

    # Read result
    with open(result_file, "r") as f:
        result = json.loads(f.read())

    # Clean up
    os.remove(result_file)
    print("✅ Received feedback from Gradio interface")

    return result["data"]