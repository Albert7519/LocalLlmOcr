import streamlit as st
import pandas as pd
from utils import helpers
import os
import json
from typing import List, Dict, Any

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

    # 处理可能无效的output_format_hint值
    valid_formats = ["CSV", "JSON", "XLSX"]
    current_format_hint = edited_data.get("output_format_hint", "CSV") # Get current hint or default to CSV

    # 规范化格式提示
    if current_format_hint not in valid_formats:
        normalized_format = "CSV" # Default fallback
        if isinstance(current_format_hint, str):
            # Try to find a valid format within the string
            if "JSON" in current_format_hint.upper():
                normalized_format = "JSON"
            elif "XLSX" in current_format_hint.upper():
                normalized_format = "XLSX"
            elif "CSV" in current_format_hint.upper():
                 normalized_format = "CSV"
        current_format_hint = normalized_format # Use the normalized format

    # 确保格式有效
    if current_format_hint not in valid_formats:
        current_format_hint = "CSV" # Final safety net

    try:
        selected_index = valid_formats.index(current_format_hint)
    except ValueError:
        selected_index = 0 # Default to CSV if index lookup fails unexpectedly

    # 使用验证过的索引
    edited_data["output_format_hint"] = st.selectbox(
        "建议输出格式",
        options=valid_formats,
        index=selected_index # Use the validated index
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
            
            # 为了避免状态冲突，使用一个唯一键，比如基于时间或模板名称
            save_key = f"save_as_name_{selected_option.replace(' ', '_')}"
            new_template_name = st.text_input("另存为新模板名称 (留空则不保存)", key=save_key).strip()
            
            if st.button("保存模板", key=f"save_template_button_{selected_option.replace(' ', '_')}"):
                if new_template_name:
                    template_to_save = edited_template # Save the latest edits
                    if helpers.save_template(new_template_name, template_to_save):
                        st.success(f"模板 '{new_template_name}' 已保存。请在上方重新选择以使用。")
                        # 不尝试直接修改input值，而是通过重新运行来刷新界面
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

# 添加智能推荐模板相关函数
def analyze_image_content(image_data: Dict[str, Any], model, processor) -> Dict[str, Any]:
    """使用大模型分析图片内容，返回推荐的模板字段"""
    from modules import processor as processor_module  # 避免循环导入
    
    if not image_data or not model or not processor:
        st.error("无法分析图片内容，缺少必要参数。")
        return None
    
    # 提示词设计：让模型分析图片并提取可能的字段和结构
    prompt = """你是一个专业的文档分析专家。请分析这张图片，识别出其中包含的所有关键信息字段。

任务要求：
1. 分析图片中包含的所有文本内容
2. 识别出所有可能作为数据字段的信息（如姓名、日期、金额、编号等）
3. 对每个识别出的字段，提供字段名称、数据格式和是否为必要字段
4. 将识别结果以JSON格式输出

输出格式示例：
```json
{
  "fields": [
    {"name": "字段1", "format": "文本/数字/日期等", "required": true/false},
    {"name": "字段2", "format": "文本/数字/日期等", "required": true/false}
  ],
  "document_type": "发票/报表/证件/其他文档类型",
  "output_format_hint": "CSV/JSON/XLSX"
}
```

请确保以上JSON格式严格准确，不要添加额外解释，因为这将用于程序自动解析。"""

    try:
        # 使用现有处理模块调用大模型分析图片
        analysis_result = processor_module.process_images(
            images=[image_data],
            template={"fields": [], "output_format_hint": "JSON", "notes": "自动分析图片内容"},
            model=model,
            processor=processor,
            custom_prompt=prompt,
            single_image_mode=True
        )
        
        if not analysis_result or not analysis_result[0].get('raw_output'):
            return None
            
        # 尝试解析JSON输出
        raw_output = analysis_result[0]['raw_output']
        
        # 清理输出中的代码块标记
        clean_output = raw_output.strip()
        if "```json" in clean_output:
            clean_output = clean_output.split("```json")[1]
        if "```" in clean_output:
            clean_output = clean_output.split("```")[0]
        clean_output = clean_output.strip()
        
        try:
            template_data = json.loads(clean_output)
            # 添加基本模板信息
            if "document_type" in template_data:
                doc_type = template_data["document_type"]
            else:
                doc_type = "自动识别的文档"
                
            template = {
                "name": f"AI推荐: {doc_type}",
                "description": f"由AI自动分析图像内容生成的{doc_type}模板",
                "fields": template_data.get("fields", []),
                "output_format_hint": template_data.get("output_format_hint", "CSV"),
                "notes": "此模板由AI自动推荐生成，请根据需要调整字段。"
            }
            return template
        except json.JSONDecodeError as e:
            st.error(f"无法解析模型输出为JSON: {e}")
            return None
            
    except Exception as e:
        st.error(f"分析图片时出错: {e}")
        return None

def handle_ai_recommendation(image_data: Dict[str, Any], model, processor):
    """处理AI推荐模板的流程"""
    if not image_data:
        st.sidebar.warning("请先上传图片以使用AI推荐功能。")
        return None
        
    # 修改：将spinner从侧边栏移到主页面区域
    st.sidebar.info("🤖 AI正在分析图片内容...")
    with st.spinner("正在分析图片内容，请稍候..."):
        recommended_template = analyze_image_content(image_data, model, processor)
        
    if recommended_template:
        # 保存临时模板并返回ID
        temp_id = helpers.save_temp_template(recommended_template)
        if temp_id:
            st.sidebar.success("✅ AI已成功分析图片并推荐模板！")
            # 设置会话状态，使得模板选择器自动选择这个临时模板
            # 获取当前选项列表
            available_templates = helpers.list_available_templates()
            options = ["--- 选择一个模板 ---"] + available_templates + ["手动创建/编辑新模板"]
            if temp_id in available_templates:
                st.session_state.selected_template_option = temp_id
                return temp_id
        else:
            st.sidebar.error("无法保存AI推荐的模板。")
            
    return None
