import streamlit as st
import pandas as pd # Add pandas import
# Remove old direct imports of torch, modelscope, PIL, io if they are no longer directly used here
# Keep streamlit import

# Import modules
from modules import auth, file_handler, preprocessor, processor, formatter, model_loader
from utils import helpers # Ensure helpers is imported if needed directly

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
            # Convert processed_images list to a dictionary with filename as key
            input_images_dict = {img_data['name']: img_data['image'] for img_data in processed_images}
            st.session_state.input_images = input_images_dict # Store as dictionary

            # Call the processor module (pass the original list of dicts)
            results = processor.process_images(
                images=processed_images, # Pass the list as expected by processor
                template=active_template,
                model=model,
                processor=processor_instance
            )
            # Store results in session state
            st.session_state.processing_results = results

            # Format data immediately
            formatted_data = formatter.format_data_for_export(results, active_template)
            st.session_state.formatted_data = formatted_data
            # Initialize or clear edited data
            st.session_state.edited_data = {}
    elif not processed_images:
        st.warning("请先上传图片文件。")
    elif not active_template:
        st.warning("请选择或创建一个有效的处理模板。")

# Display results if they exist
if 'processing_results' in st.session_state:
    formatter.display_results(st.session_state.processing_results) # Step 3

# Add editable data section if formatted data exists
if 'formatted_data' in st.session_state and st.session_state.formatted_data:
    st.subheader("4. 核查与修改提取结果")
    st.info("您可以在下方的表格中直接修改提取的数据。修改后的数据将用于最终下载。点击“查看原图”可展开对应图片。")

    # Initialize edited_data if it doesn't exist
    if 'edited_data' not in st.session_state:
        st.session_state.edited_data = {}

    # Retrieve input images dictionary from session state
    # Default to empty dict if not found
    input_images_dict = st.session_state.get('input_images', {})

    for filename, data in st.session_state.formatted_data.items():
        st.markdown(f"---") # Separator for each file section
        st.markdown(f"**文件: {filename}**")

        # Get the corresponding image from the dictionary
        image_data = input_images_dict.get(filename) # Now correctly uses .get() on a dictionary

        # Display image in an expander if available
        if image_data:
            with st.expander("查看原图", expanded=False):
                st.image(image_data, caption=f"原图: {filename}", use_container_width=True)
        else:
            st.caption("未找到对应的预览图。")

        # Display the data editor or other data representations
        if isinstance(data, pd.DataFrame) and not data.empty:
            edited_df = st.data_editor(
                data,
                key=f"editor_{filename}",
                num_rows="dynamic"
            )
            st.session_state.edited_data[filename] = edited_df
        elif isinstance(data, pd.DataFrame) and data.empty:
            st.write("此文件未提取到表格数据。")
            st.session_state.edited_data[filename] = data
        elif isinstance(data, str) and data.startswith("错误:"):
             st.error(f"处理错误，无法编辑: {data}")
             st.session_state.edited_data[filename] = data
        elif isinstance(data, dict):
             st.write("数据为JSON对象，暂不支持直接编辑。")
             st.json(data)
             st.session_state.edited_data[filename] = data
        else:
             st.write("数据格式未知或无法编辑。")
             st.text(str(data))
             st.session_state.edited_data[filename] = data


# Provide download buttons using the potentially edited data
if 'edited_data' in st.session_state and st.session_state.edited_data:
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
atexit.register(helpers.cleanup_temp_templates)
