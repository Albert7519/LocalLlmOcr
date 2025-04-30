import streamlit as st
import pandas as pd # Add pandas import
# Remove old direct imports of torch, modelscope, PIL, io if they are no longer directly used here
# Keep streamlit import

# Import modules
from modules import auth, file_handler, preprocessor, processor, formatter, model_loader

# --- Page Config ---
st.set_page_config(layout="wide", page_title="本地 LLM OCR 应用")

# --- Authentication ---
if not auth.show_login_form():
    st.stop() # Stop execution if not logged in

# --- Load Model (runs only once after login) ---
# Ensure model and processor are loaded before proceeding
model, processor_instance = model_loader.load_model_and_processor() # Renamed processor to avoid conflict

if model is None or processor_instance is None:
    st.error("无法加载模型或处理器。请检查日志或环境配置。")
    st.stop()

# --- Main Application UI ---
st.title("📄 本地 LLM 文档处理应用")

# --- Sidebar ---
with st.sidebar:
    st.header("操作面板")
    auth.add_logout_button() # Add logout button
    st.markdown("---")

    # 1. File Upload
    processed_images = file_handler.handle_file_upload()

    # 添加AI智能推荐模板按钮
    if processed_images:
        st.markdown("---")
        st.subheader("AI 智能分析")
        ai_recommend = st.button("🔍 AI 分析图片并推荐模板", key="ai_recommend_button")
        
        # 处理AI推荐请求
        if ai_recommend and processed_images:
            # 使用第一张图片进行分析
            first_image = processed_images[0]
            # 调用preprocessor的处理函数
            preprocessor.handle_ai_recommendation(first_image, model, processor_instance)
        elif ai_recommend:
            st.warning("请先上传图片以使用AI推荐功能。")

    # 2. Template Selection/Editing
    active_template = preprocessor.handle_preprocessing()

    # 3. Start Processing Button
    st.markdown("---")
    start_processing = st.button("🚀 开始处理", disabled=(not processed_images or not active_template))

# --- Main Content Area ---
if start_processing:
    if processed_images and active_template and model and processor_instance:
        with st.spinner("🧠 正在调用大模型处理图片..."):
            # Call the processor module
            results = processor.process_images(
                images=processed_images,
                template=active_template,
                model=model,
                processor=processor_instance # Pass the loaded processor instance
            )
            # Store results in session state to persist across reruns
            st.session_state.processing_results = results

            # Format data immediately after processing
            formatted_data = formatter.format_data_for_export(results, active_template)
            st.session_state.formatted_data = formatted_data
            # Initialize or clear edited data when new processing starts
            st.session_state.edited_data = {}
    elif not processed_images:
        st.warning("请先上传图片文件。")
    elif not active_template:
        st.warning("请选择或创建一个有效的处理模板。")

# Display results if they exist in session state
if 'processing_results' in st.session_state:
    formatter.display_results(st.session_state.processing_results) # Step 3

# Add editable data section if formatted data exists
if 'formatted_data' in st.session_state and st.session_state.formatted_data:
    st.subheader("4. 核查与修改提取结果")
    st.info("您可以在下方的表格中直接修改提取的数据。修改后的数据将用于最终下载。")

    # Initialize edited_data if it doesn't exist (e.g., after page reload)
    if 'edited_data' not in st.session_state:
        st.session_state.edited_data = {}

    # Use columns for better layout if many files
    # num_columns = min(len(st.session_state.formatted_data), 3) # Example: max 3 columns
    # cols = st.columns(num_columns)
    # current_col = 0

    for filename, data in st.session_state.formatted_data.items():
        # with cols[current_col]: # Assign to a column
        if isinstance(data, pd.DataFrame) and not data.empty:
            st.markdown(f"**文件: {filename}**")
            # Use filename as key for data editor state
            edited_df = st.data_editor(
                data,
                key=f"editor_{filename}", # Unique key for each editor
                num_rows="dynamic" # Allow adding/deleting rows if needed
            )
            # Store the edited dataframe in session state
            st.session_state.edited_data[filename] = edited_df
        elif isinstance(data, pd.DataFrame) and data.empty:
            st.markdown(f"**文件: {filename}**")
            st.write("此文件未提取到表格数据。")
            st.session_state.edited_data[filename] = data # Store empty df
        elif isinstance(data, str) and data.startswith("错误:"):
             st.markdown(f"**文件: {filename}**")
             st.error(f"处理错误，无法编辑: {data}")
             st.session_state.edited_data[filename] = data # Store error string
        elif isinstance(data, dict):
             st.markdown(f"**文件: {filename}**")
             st.write("数据为JSON对象，暂不支持直接编辑。")
             st.json(data) # Display JSON
             st.session_state.edited_data[filename] = data # Store original dict
        else:
             st.markdown(f"**文件: {filename}**")
             st.write("数据格式未知或无法编辑。")
             st.text(str(data))
             st.session_state.edited_data[filename] = data # Store original data

        # current_col = (current_col + 1) % num_columns # Move to next column
    st.markdown("---")


# Provide download buttons using the potentially edited data
if 'edited_data' in st.session_state and st.session_state.edited_data:
    # Pass the edited data to the download function
    formatter.provide_download_buttons(st.session_state.edited_data) # Step 5
elif 'formatted_data' in st.session_state:
     # Fallback to formatted_data if edited_data somehow doesn't exist but formatted_data does
     # This might happen on first run before editor interaction
     st.info("编辑表格后将启用下载功能。")
     # formatter.provide_download_buttons(st.session_state.formatted_data) # Or show download for unedited initially? Let's wait for edit.
else:
    # Show placeholder if no processing has been done yet
    st.info("处理完成后，将在此处显示结果预览、编辑和下载选项。")

# Optional: Display uploaded image previews (can be simple names or thumbnails)
if processed_images:
     with st.expander("查看已上传图片"):
         for img_data in processed_images:
             st.write(f"- {img_data['name']}")
             # Optionally display small thumbnails
             # st.image(img_data['image'], width=100)

# 添加对临时模板的清理逻辑（应用关闭时）
import atexit
from utils import helpers

atexit.register(helpers.cleanup_temp_templates)
