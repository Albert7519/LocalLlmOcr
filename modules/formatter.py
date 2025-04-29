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
    try:
        # Use StringIO to treat the string as a file
        # Assume no header in the raw output as per prompt instructions
        data = StringIO(raw_output.strip())
        df = pd.read_csv(data, header=None, names=field_names, skipinitialspace=True)
        # Attempt basic type conversion based on template hints (best effort)
        for field in template.get('fields', []):
            col_name = field.get('name')
            col_format = field.get('format', '').lower()
            if col_name in df.columns:
                if '数字' in col_format or 'number' in col_format or 'int' in col_format or 'float' in col_format:
                    df[col_name] = pd.to_numeric(df[col_name], errors='coerce') # Coerce errors to NaN
                elif 'date' in col_format or '日期' in col_format:
                     df[col_name] = pd.to_datetime(df[col_name], errors='coerce') # Coerce errors to NaT
        return df
    except Exception as e:
        st.warning(f"无法将输出解析为CSV: {e}. 将作为单列文本处理。")
        # Fallback: return as a single column DataFrame
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
