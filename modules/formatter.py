import streamlit as st
import pandas as pd
import json
from io import StringIO, BytesIO
from typing import List, Dict, Any
import os # Import os for path manipulation

def display_results(results: List[Dict[str, Any]]):
    """Displays the raw processing results from the LLM."""
    st.subheader("3. 处理结果预览") # Changed section number
    if not results:
        st.info("没有可显示的处理结果。请先上传文件并运行处理。")
        return

    for result in results:
        st.markdown(f"**文件: {result.get('name', '未知文件')}**")
        if result.get('error'):
            st.error(f"处理错误: {result.get('raw_output', '未知错误')}")
        else:
            # Use unique key for each text_area based on filename
            st.text_area(f"原始输出##{result.get('name', 'raw')}", value=result.get('raw_output', ''), height=150, key=f"raw_output_{result.get('name')}")
        # Optionally display the prompt used (for debugging)
        with st.expander("查看使用的提示词"):
            st.text(result.get('prompt', '无提示词信息'))
        st.markdown("---")

def _parse_csv_like(raw_output: str, template: Dict[str, Any]) -> pd.DataFrame:
    """Attempts to parse CSV-like text output into a DataFrame."""
    field_names = [field.get('name', f'列{i+1}') for i, field in enumerate(template.get('fields', []))]
    
    # 检查输出是否包含表格形式的数据（带有 | 分隔符）
    if '|' in raw_output and '-|-' in raw_output.replace(' ', ''):
        # 处理Markdown表格格式
        try:
            # 分割行
            lines = [line.strip() for line in raw_output.split('\n') if line.strip()]
            
            # 移除可能的标题行和表头分隔行
            data_lines = []
            for line in lines:
                # 跳过表头分隔行 (通常包含 -|-|- 这样的分隔符)
                if line.replace(' ', '').startswith('|') and all(c == '-' or c == '|' or c == ':' or c == ' ' for c in line):
                    continue
                # 保留数据行
                if '|' in line:
                    # 清理行数据、分割并去除空格
                    cleaned_line = line.strip('| ')
                    cells = [cell.strip() for cell in cleaned_line.split('|')]
                    data_lines.append(cells)
            
            # 如果第一行看起来像表头（与字段名匹配度高），就跳过
            if data_lines and len(data_lines) > 1:
                header_check = data_lines[0]
                if len(header_check) == len(field_names):
                    # 检查是否为表头（与字段名称有较高匹配度）
                    match_count = sum(1 for h, f in zip(header_check, field_names) if h.lower().strip() == f.lower().strip())
                    if match_count >= len(field_names) * 0.5:  # 如果50%以上匹配，认为是表头
                        data_lines = data_lines[1:]  # 跳过表头
            
            # 创建DataFrame
            if data_lines:
                # 检查列数是否与字段数匹配，处理不一致情况
                column_count = max(len(row) for row in data_lines)
                if column_count > len(field_names):
                    # 如果实际列数比字段数多，增加字段名
                    field_names.extend([f'未命名列{i+1}' for i in range(column_count - len(field_names))])
                elif column_count < len(field_names):
                    # 如果实际列数比字段数少，使用实际列数
                    field_names = field_names[:column_count]
                
                # Read data as string first to preserve formatting
                df = pd.DataFrame(data_lines, columns=field_names).astype(str)

                # 数据清理：去除千位分隔符和货币符号等
                for col in df.columns:
                    if df[col].dtype == 'object':
                        # 清理千位分隔符、货币符号等
                        df[col] = df[col].str.replace(',', '', regex=True)
                        df[col] = df[col].str.replace('¥', '', regex=True)
                        df[col] = df[col].str.replace('$', '', regex=True)
                        df[col] = df[col].str.replace('￥', '', regex=True)
                
                # Attempt type conversion based on template hints, skipping text-like formats
                for field in template.get('fields', []):
                    col_name = field.get('name')
                    col_format = field.get('format', '').lower()
                    if col_name in df.columns:
                        # Check if format hint suggests it should REMAIN text
                        is_text_like = any(hint in col_format for hint in ['文本', '编号', '代码', 'id', 'text', 'code', 'string'])

                        if not is_text_like: # Only convert if not explicitly text-like
                            if '数字' in col_format or 'number' in col_format or 'int' in col_format or 'float' in col_format or '金额' in col_format:
                                # Convert to numeric, coercing errors. Keep as object if conversion fails.
                                df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
                                # Optional: Fill NaNs created by coerce with original string or empty string
                                # df[col_name] = df[col_name].fillna('') # Or keep as NaN depending on desired output
                            elif 'date' in col_format or '日期' in col_format:
                                # Convert to datetime, coercing errors. Keep as object if conversion fails.
                                df[col_name] = pd.to_datetime(df[col_name], errors='coerce')
                                # Optional: Format date string after conversion if needed
                                # df[col_name] = df[col_name].dt.strftime('%Y-%m-%d') # Example format

                return df
        except Exception as e:
            st.warning(f"解析表格时出错: {e}，将尝试其他解析方法。")
    
    # 如果不是表格或解析失败，尝试CSV解析
    try:
        # 移除可能的Markdown标记和非CSV内容
        cleaned_output = raw_output
        # 如果找到代码块，提取其内容
        if '```' in cleaned_output:
            blocks = cleaned_output.split('```')
            for i, block in enumerate(blocks):
                if i % 2 == 1 and ('csv' in block.lower() or ',' in block):
                    cleaned_output = block
                    break
        
        # 使用StringIO处理CSV格式，READ ALL AS STRING INITIALLY
        data = StringIO(cleaned_output.strip())
        # Read CSV with all columns as string type initially
        df = pd.read_csv(data, header=None, names=field_names, skipinitialspace=True, dtype=str)

        # Attempt type conversion based on template hints, skipping text-like formats
        for field in template.get('fields', []):
            col_name = field.get('name')
            col_format = field.get('format', '').lower()
            if col_name in df.columns:
                 # Check if format hint suggests it should REMAIN text
                is_text_like = any(hint in col_format for hint in ['文本', '编号', '代码', 'id', 'text', 'code', 'string'])

                if not is_text_like: # Only convert if not explicitly text-like
                    if '数字' in col_format or 'number' in col_format or 'int' in col_format or 'float' in col_format or '金额' in col_format:
                        # Convert to numeric, coercing errors. Keep as object if conversion fails.
                        df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
                        # Optional: Fill NaNs created by coerce
                        # df[col_name] = df[col_name].fillna('')
                    elif 'date' in col_format or '日期' in col_format:
                         # Convert to datetime, coercing errors. Keep as object if conversion fails.
                        df[col_name] = pd.to_datetime(df[col_name], errors='coerce')
                        # Optional: Format date string
                        # df[col_name] = df[col_name].dt.strftime('%Y-%m-%d')

        return df
    except Exception as e:
        st.warning(f"无法将输出解析为CSV: {e}. 将作为单列文本处理。")
        # Fallback: return single-column DataFrame, ensure dtype is string
        return pd.DataFrame({'原始输出': raw_output.strip().split('\n')}).astype(str)


def _parse_json_like(raw_output: str, template: Dict[str, Any]) -> pd.DataFrame | Dict: # Added template parameter
    """Attempts to parse JSON-like text output."""
    try:
        # Clean potential markdown code blocks
        cleaned_output = raw_output.strip().removeprefix('```json').removesuffix('```').strip()
        data = json.loads(cleaned_output)
        df = None
        if isinstance(data, list): # List of records
            df = pd.DataFrame(data)
        elif isinstance(data, dict): # Single record or structured dict
             try:
                 df = pd.DataFrame([data])
             except ValueError:
                 return data # Return the dict itself if not suitable for DataFrame

        if df is not None:
            # Ensure columns intended as text remain string type based on template
            if template and 'fields' in template:
                for field in template.get('fields', []):
                    col_name = field.get('name')
                    col_format = field.get('format', '').lower()
                    if col_name in df.columns:
                        is_text_like = any(hint in col_format for hint in ['文本', '编号', '代码', 'id', 'text', 'code', 'string'])
                        # Also treat columns as text if they contain non-numeric strings after initial load
                        # This handles cases where template might be missing/inaccurate but data is clearly text
                        try:
                            if not is_text_like and df[col_name].apply(lambda x: isinstance(x, str) and not x.replace('.', '', 1).isdigit() and x).any():
                                is_text_like = True
                        except Exception: # Handle potential errors during apply
                            pass

                        if is_text_like:
                            # Explicitly convert to string to preserve format
                            df[col_name] = df[col_name].astype(str)
                        # Optional: Add selective numeric/date conversion here if needed, similar to _parse_csv_like
                        # else:
                        #    if '数字' in col_format ...:
                        #        df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
                        #    elif '日期' in col_format ...:
                        #        df[col_name] = pd.to_datetime(df[col_name], errors='coerce')

            return df # Return df after specific conversions, not converting everything to str here.
        else: # Handle cases where JSON wasn't list or dict suitable for DataFrame
            st.warning("解析的JSON不是列表或字典，将作为文本处理。")
            return pd.DataFrame({'原始输出': [cleaned_output]}).astype(str) # Ensure string type

    except json.JSONDecodeError as e:
        st.warning(f"无法将输出解析为JSON: {e}. 将作为单列文本处理。")
        return pd.DataFrame({'原始输出': raw_output.strip().split('\n')}).astype(str) # Ensure string type
    except Exception as e:
        st.warning(f"处理JSON输出时出错: {e}. 将作为单列文本处理。")
        return pd.DataFrame({'原始输出': raw_output.strip().split('\n')}).astype(str) # Ensure string type


def format_data_for_export(
    results: List[Dict[str, Any]],
    template: Dict[str, Any]
) -> Dict[str, pd.DataFrame | Dict | str]:
    """
    Formats the raw results based on the template's output hint.
    Returns a dictionary where keys are filenames and values are DataFrames, dicts, or raw strings.
    """
    formatted_data = {}
    if not results or not template:
        return formatted_data

    output_hint = template.get('output_format_hint', 'CSV').upper() # Default to CSV

    for result in results:
        filename = result.get('name', 'unknown_file')
        raw_output = result.get('raw_output', '')
        if result.get('error'):
            formatted_data[filename] = f"错误: {raw_output}" # Store error message
            continue

        if not raw_output.strip():
             formatted_data[filename] = pd.DataFrame() # Empty DataFrame for empty output
             continue

        parsed_result = None
        if output_hint == 'CSV':
            # Pass template to parsing function
            parsed_result = _parse_csv_like(raw_output, template)
        elif output_hint == 'JSON':
             # Pass template to parsing function
            parsed_result = _parse_json_like(raw_output, template)
        # Add XLSX hint handling if needed, often similar to CSV parsing
        elif output_hint == 'XLSX':
             # Pass template to parsing function
             parsed_result = _parse_csv_like(raw_output, template) # Treat like CSV for parsing
        else: # Default or unknown hint, treat as raw text split by lines
            parsed_result = pd.DataFrame({'原始输出': raw_output.strip().split('\n')}).astype(str) # Ensure string type

        formatted_data[filename] = parsed_result

    return formatted_data


def provide_download_buttons(edited_data: Dict[str, pd.DataFrame | Dict | str]): # Changed parameter name
    """Provides download buttons for the formatted and potentially edited data."""
    st.subheader("5. 下载提取结果") # Changed section number

    if not edited_data: # Check the edited_data parameter
        st.info("没有可供下载的数据。")
        return

    # Combine data if possible (e.g., multiple CSVs into one)
    # Use the edited_data passed to the function
    all_dfs = [df for df in edited_data.values() if isinstance(df, pd.DataFrame) and not df.empty]
    combined_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

    # --- Offer Combined Download ---
    if not combined_df.empty:
        st.write("**合并下载 (所有成功处理并编辑后的表格数据):**") # Updated description
        col1, col2, col3 = st.columns(3)

        # CSV Download
        # Ensure all data is string for CSV to prevent Excel auto-formatting issues
        csv_data = combined_df.astype(str).to_csv(index=False).encode('utf-8-sig') # Use utf-8-sig for Excel compatibility
        col1.download_button(
            label="下载合并 CSV",
            data=csv_data,
            file_name="combined_output_edited.csv", # Indicate edited
            mime="text/csv",
            key="combined_csv_download_edited" # Unique key
        )

        # XLSX Download
        output = BytesIO()
        # Write to Excel, explicitly setting text format for all columns might be needed
        # if openpyxl auto-detection is still causing issues.
        # However, pandas usually handles this well if dtypes are strings.
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
             # Convert to string before writing to Excel to preserve formatting
            combined_df.astype(str).to_excel(writer, index=False, sheet_name='Combined Data')
        xlsx_data = output.getvalue()
        col2.download_button(
            label="下载合并 XLSX",
            data=xlsx_data,
            file_name="combined_output_edited.xlsx", # Indicate edited
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="combined_xlsx_download_edited" # Unique key
        )

        # JSON Download (List of Records)
        # Convert to string before JSON dump might not be necessary, JSON handles types
        json_data = combined_df.to_json(orient="records", indent=2, force_ascii=False)
        col3.download_button(
            label="下载合并 JSON",
            data=json_data.encode('utf-8'), # Encode JSON string to bytes
            file_name="combined_output_edited.json", # Indicate edited
            mime="application/json",
            key="combined_json_download_edited" # Unique key
        )
        st.markdown("---")


    # --- Offer Individual File Downloads ---
    st.write("**单独下载 (每个文件，包含编辑后的数据):**") # Updated description
    # Use the edited_data passed to the function
    for filename, data in edited_data.items():
        base_name = os.path.splitext(filename)[0] # Remove original extension
        with st.expander(f"下载选项: {filename}"):
            if isinstance(data, pd.DataFrame):
                if not data.empty:
                    col1_ind, col2_ind, col3_ind = st.columns(3)
                    # CSV - Convert to string
                    csv_ind = data.astype(str).to_csv(index=False).encode('utf-8-sig')
                    col1_ind.download_button(f"下载 CSV", csv_ind, f"{base_name}_output_edited.csv", "text/csv", key=f"csv_edited_{filename}")
                    # XLSX - Convert to string
                    output_ind = BytesIO()
                    with pd.ExcelWriter(output_ind, engine='openpyxl') as writer:
                        data.astype(str).to_excel(writer, index=False, sheet_name='Sheet1')
                    xlsx_ind = output_ind.getvalue()
                    col2_ind.download_button(f"下载 XLSX", xlsx_ind, f"{base_name}_output_edited.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=f"xlsx_edited_{filename}")
                    # JSON
                    json_ind = data.to_json(orient="records", indent=2, force_ascii=False)
                    col3_ind.download_button(f"下载 JSON", json_ind.encode('utf-8'), f"{base_name}_output_edited.json", "application/json", key=f"json_edited_{filename}")
                else:
                    st.info("此文件无表格数据可下载。")
            elif isinstance(data, dict): # Handle dictionary data (e.g., from JSON parsing)
                 json_str = json.dumps(data, indent=2, ensure_ascii=False)
                 st.download_button(f"下载 JSON", json_str.encode('utf-8'), f"{base_name}_output.json", "application/json", key=f"json_dict_{filename}") # No edit indication needed
                 st.write("数据为JSON对象，仅提供JSON下载。")
            elif isinstance(data, str) and data.startswith("错误:"):
                 st.error(f"无法下载，处理时发生错误: {data}")
            else: # Assume raw string or other format
                 st.download_button(f"下载原始文本", str(data).encode('utf-8'), f"{base_name}_output.txt", "text/plain", key=f"txt_{filename}") # No edit indication needed
                 st.write("数据格式无法识别为表格，提供原始文本下载。")

def add_logout_button():
    """Adds a logout button to the sidebar if logged in."""
    if st.session_state.get('logged_in', False):
        if st.sidebar.button("退出登录"):
            st.session_state.logged_in = False
            # Clear other relevant session state if needed upon logout
            # Example: Clear results, selected template, formatted and edited data
            keys_to_clear = [
                'processing_results', 'formatted_data', 'edited_data', # Added edited_data
                'selected_template_option', 'manual_template',
                'loaded_template', 'loaded_template_name', 'edited_loaded_template'
            ]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun() # Rerun to show the login form again
