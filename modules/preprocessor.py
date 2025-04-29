import streamlit as st
import pandas as pd
from utils import helpers # Import helper functions for templates
import os # Needed for checking template existence

# Default structure for a new template
DEFAULT_TEMPLATE = {
  "name": "新模板",
  "description": "自定义提取规则。",
  "fields": [
    {"name": "字段1", "format": "文本", "required": True},
    {"name": "字段2", "format": "数字", "required": False},
  ],
  "output_format_hint": "CSV",
  "notes": ""
}

def show_template_selection():
    """Displays template selection UI in the sidebar."""
    st.sidebar.subheader("1. 选择或创建提取模板")
    available_templates = helpers.list_available_templates()
    options = ["--- 选择一个模板 ---"] + available_templates + ["手动创建/编辑新模板"]

    # Use session state to remember the selection
    if 'selected_template_option' not in st.session_state:
        st.session_state.selected_template_option = options[0] # Default to placeholder

    selected_option = st.sidebar.selectbox(
        "选择预设模板或手动创建:",
        options,
        key='selected_template_option' # Persist selection
    )
    return selected_option

def edit_template_interactive(template_data: dict) -> dict:
    """Provides an interactive UI to edit template details."""
    edited_data = template_data.copy() # Work on a copy

    edited_data["name"] = st.text_input("模板名称", value=edited_data.get("name", "新模板"))
    edited_data["description"] = st.text_area("模板描述", value=edited_data.get("description", ""))

    st.markdown("**编辑提取字段:**")
    # Convert fields list to DataFrame for st.data_editor
    if "fields" not in edited_data or not isinstance(edited_data["fields"], list):
        edited_data["fields"] = [] # Ensure fields is a list

    fields_df = pd.DataFrame(edited_data["fields"])

    # Ensure required columns exist, even if empty
    for col in ["name", "format", "required"]:
         if col not in fields_df.columns:
             fields_df[col] = None if col != 'required' else False # Default required to False

    # Reorder columns for display
    display_columns = ["name", "format", "required"] + [col for col in fields_df.columns if col not in ["name", "format", "required"]]
    fields_df = fields_df[display_columns]


    edited_fields_df = st.data_editor(
        fields_df,
        num_rows="dynamic", # Allow adding/deleting rows
        column_config={
            "name": st.column_config.TextColumn("字段名称 (必填)", required=True),
            "format": st.column_config.TextColumn("格式提示 (可选)"),
            "required": st.column_config.CheckboxColumn("是否必须?", default=False),
        },
        key="template_fields_editor" # Unique key for the editor
    )

    # Convert DataFrame back to list of dicts, handling potential NaNs/None
    edited_data["fields"] = edited_fields_df.astype(object).where(pd.notnull(edited_fields_df), None).to_dict('records')


    edited_data["output_format_hint"] = st.selectbox(
        "建议输出格式",
        options=["CSV", "JSON", "XLSX"],
        index=["CSV", "JSON", "XLSX"].index(edited_data.get("output_format_hint", "CSV"))
    )
    edited_data["notes"] = st.text_area("模板备注", value=edited_data.get("notes", ""))

    return edited_data

def handle_preprocessing():
    """Manages template selection, loading, editing, and saving."""
    selected_option = show_template_selection()
    current_template = None

    if selected_option == "--- 选择一个模板 ---":
        st.sidebar.info("请选择一个模板或选择手动创建。")
        return None # No template selected yet
    elif selected_option == "手动创建/编辑新模板":
        # Use session state to store the manually created template if it doesn't exist
        if 'manual_template' not in st.session_state:
             st.session_state.manual_template = DEFAULT_TEMPLATE.copy()
        current_template = st.session_state.manual_template
        st.sidebar.write("当前模式：手动创建/编辑")
    else:
        # Load selected template
        # Check if the loaded template matches the selection, otherwise reload
        if 'loaded_template_name' not in st.session_state or st.session_state.loaded_template_name != selected_option:
            st.session_state.loaded_template = helpers.load_template(selected_option)
            st.session_state.loaded_template_name = selected_option # Store the name of the loaded template
        current_template = st.session_state.get('loaded_template', None)


    if current_template:
        st.sidebar.subheader("2. (可选) 微调模板")
        with st.sidebar.expander("展开以编辑当前模板", expanded=(selected_option == "手动创建/编辑新模板")): # Expand if manual
            edited_template = edit_template_interactive(current_template)

            # Update the template in session state immediately after editing
            if selected_option == "手动创建/编辑新模板":
                st.session_state.manual_template = edited_template
                current_template = edited_template # Ensure current_template reflects edits
            else:
                 # If editing a loaded template, store the *edited* version separately
                 # to avoid overwriting the original loaded one until save
                 st.session_state.edited_loaded_template = edited_template
                 current_template = edited_template # Use the edited version going forward

            st.markdown("---") # Separator
            st.write("**保存模板:**")
            new_template_name = st.text_input("另存为新模板名称 (留空则不保存)", key="save_as_name").strip()
            if st.button("保存模板", key="save_template_button"): # Added key
                if new_template_name:
                    template_to_save = edited_template # Save the latest edits
                    if helpers.save_template(new_template_name, template_to_save):
                        st.success(f"模板 '{new_template_name}' 已保存。请在上方重新选择以使用。")
                        # Clear the input field after saving
                        st.session_state.save_as_name = ""
                        # Force rerun to update template list
                        st.rerun()
                    else:
                        st.error("保存失败。")
                else:
                    st.warning("请输入新模板的名称。")

        # Return the *currently active* template (either original loaded, edited loaded, or manual)
        return current_template
    else:
        if selected_option != "--- 选择一个模板 ---": # Avoid error message if nothing is selected
             st.sidebar.error(f"无法加载模板: {selected_option}")
        return None
