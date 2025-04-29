import streamlit as st
import pandas as pd
import json
from io import StringIO, BytesIO
from typing import List, Dict, Any
import os # Import os for path manipulation

def display_results(results: List[Dict[str, Any]]):
    """Displays the raw processing results from the LLM."""
    st.subheader("3. 处理结果预览")
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
                
                df = pd.DataFrame(data_lines, columns=field_names)
                
                # 数据清理：去除千位分隔符和货币符号等
                for col in df.columns:
                    if df[col].dtype == 'object':
                        # 清理千位分隔符、货币符号等
                        df[col] = df[col].str.replace(',', '', regex=True)
                        df[col] = df[col].str.replace('¥', '', regex=True)
                        df[col] = df[col].str.replace('$', '', regex=True)
                        df[col] = df[col].str.replace('￥', '', regex=True)
                
                # 尝试将数值列转换为数值类型
                for field in template.get('fields', []):
                    col_name = field.get('name')
                    col_format = field.get('format', '').lower()
                    if col_name in df.columns:
                        if '数字' in col_format or 'number' in col_format or 'int' in col_format or 'float' in col_format:
                            df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
                        elif 'date' in col_format or '日期' in col_format:
                            df[col_name] = pd.to_datetime(df[col_name], errors='coerce')
                
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
        
        # 使用StringIO处理CSV格式
        data = StringIO(cleaned_output.strip())
        df = pd.read_csv(data, header=None, names=field_names, skipinitialspace=True)
        
        # 尝试基于模板提示进行数据类型转换
        for field in template.get('fields', []):
            col_name = field.get('name')
            col_format = field.get('format', '').lower()
            if col_name in df.columns:
                if '数字' in col_format or 'number' in col_format or 'int' in col_format or 'float' in col_format:
                    df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
                elif 'date' in col_format or '日期' in col_format:
                    df[col_name] = pd.to_datetime(df[col_name], errors='coerce')
        
        return df
    except Exception as e:
        st.warning(f"无法将输出解析为CSV: {e}. 将作为单列文本处理。")
        # 最后的回退方案：返回单列DataFrame
        return pd.DataFrame({'原始输出': raw_output.strip().split('\n')})


def _parse_json_like(raw_output: str) -> pd.DataFrame | Dict:
    """Attempts to parse JSON-like text output."""
    try:
        # Clean potential markdown code blocks
        cleaned_output = raw_output.strip().removeprefix('```json').removesuffix('```').strip()
        data = json.loads(cleaned_output)
        if isinstance(data, list): # List of records
            return pd.DataFrame(data)
        elif isinstance(data, dict): # Single record or structured dict
             # Try converting single dict to DataFrame row
             try:
                 return pd.DataFrame([data])
             except ValueError: # If dict structure isn't suitable for DataFrame directly
                 return data # Return the dict itself
        else:
            st.warning("解析的JSON不是列表或字典，将作为文本处理。")
            return pd.DataFrame({'原始输出': [cleaned_output]})
    except json.JSONDecodeError as e:
        st.warning(f"无法将输出解析为JSON: {e}. 将作为单列文本处理。")
        return pd.DataFrame({'原始输出': raw_output.strip().split('\n')})
    except Exception as e:
        st.warning(f"处理JSON输出时出错: {e}. 将作为单列文本处理。")
        return pd.DataFrame({'原始输出': raw_output.strip().split('\n')})


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
            parsed_result = _parse_csv_like(raw_output, template)
        elif output_hint == 'JSON':
            parsed_result = _parse_json_like(raw_output)
        # Add XLSX hint handling if needed, often similar to CSV parsing
        elif output_hint == 'XLSX':
             parsed_result = _parse_csv_like(raw_output, template) # Treat like CSV for parsing
        else: # Default or unknown hint, treat as raw text split by lines
            parsed_result = pd.DataFrame({'原始输出': raw_output.strip().split('\n')})

        formatted_data[filename] = parsed_result

    return formatted_data


def provide_download_buttons(formatted_data: Dict[str, pd.DataFrame | Dict | str]):
    """Provides download buttons for the formatted data."""
    st.subheader("4. 下载提取结果")

    if not formatted_data:
        st.info("没有可供下载的数据。")
        return

    # Combine data if possible (e.g., multiple CSVs into one)
    all_dfs = [df for df in formatted_data.values() if isinstance(df, pd.DataFrame) and not df.empty]
    combined_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

    # --- Offer Combined Download ---
    if not combined_df.empty:
        st.write("**合并下载 (所有成功处理的表格数据):**")
        col1, col2, col3 = st.columns(3)

        # CSV Download
        csv_data = combined_df.to_csv(index=False).encode('utf-8-sig') # Use utf-8-sig for Excel compatibility
        col1.download_button(
            label="下载合并 CSV",
            data=csv_data,
            file_name="combined_output.csv",
            mime="text/csv",
            key="combined_csv_download"
        )

        # XLSX Download
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            combined_df.to_excel(writer, index=False, sheet_name='Combined Data')
        xlsx_data = output.getvalue()
        col2.download_button(
            label="下载合并 XLSX",
            data=xlsx_data,
            file_name="combined_output.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="combined_xlsx_download"
        )

        # JSON Download (List of Records)
        json_data = combined_df.to_json(orient="records", indent=2, force_ascii=False)
        col3.download_button(
            label="下载合并 JSON",
            data=json_data.encode('utf-8'), # Encode JSON string to bytes
            file_name="combined_output.json",
            mime="application/json",
            key="combined_json_download"
        )
        st.markdown("---")


    # --- Offer Individual File Downloads ---
    st.write("**单独下载 (每个文件):**")
    for filename, data in formatted_data.items():
        base_name = os.path.splitext(filename)[0] # Remove original extension
        with st.expander(f"下载选项: {filename}"):
            if isinstance(data, pd.DataFrame):
                if not data.empty:
                    col1_ind, col2_ind, col3_ind = st.columns(3)
                    # CSV
                    csv_ind = data.to_csv(index=False).encode('utf-8-sig')
                    col1_ind.download_button(f"下载 CSV", csv_ind, f"{base_name}_output.csv", "text/csv", key=f"csv_{filename}")
                    # XLSX
                    output_ind = BytesIO()
                    with pd.ExcelWriter(output_ind, engine='openpyxl') as writer:
                        data.to_excel(writer, index=False, sheet_name='Sheet1')
                    xlsx_ind = output_ind.getvalue()
                    col2_ind.download_button(f"下载 XLSX", xlsx_ind, f"{base_name}_output.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=f"xlsx_{filename}")
                    # JSON
                    json_ind = data.to_json(orient="records", indent=2, force_ascii=False)
                    col3_ind.download_button(f"下载 JSON", json_ind.encode('utf-8'), f"{base_name}_output.json", "application/json", key=f"json_{filename}")
                else:
                    st.info("此文件无表格数据可下载。")
            elif isinstance(data, dict): # Handle dictionary data (e.g., from JSON parsing)
                 json_str = json.dumps(data, indent=2, ensure_ascii=False)
                 st.download_button(f"下载 JSON", json_str.encode('utf-8'), f"{base_name}_output.json", "application/json", key=f"json_dict_{filename}")
                 st.write("数据为JSON对象，仅提供JSON下载。")
            elif isinstance(data, str) and data.startswith("错误:"):
                 st.error(f"无法下载，处理时发生错误: {data}")
            else: # Assume raw string or other format
                 st.download_button(f"下载原始文本", str(data).encode('utf-8'), f"{base_name}_output.txt", "text/plain", key=f"txt_{filename}")
                 st.write("数据格式无法识别为表格，提供原始文本下载。")
