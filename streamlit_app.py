import streamlit as st
import pandas as pd
import torch # Add torch import
import gc # Add gc import

# Import modules
from modules import auth, file_handler, preprocessor, processor, formatter, model_loader
from utils import helpers

# --- Page Config ---
st.set_page_config(layout="wide", page_title="本地 LLM OCR 应用")

# --- Authentication ---
if not auth.show_login_form():
    st.stop()

# --- Initialize Session State for Model ---
if 'selected_model_name' not in st.session_state:
    st.session_state.selected_model_name = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'processor_instance' not in st.session_state:
    st.session_state.processor_instance = None

# --- Model Selection UI (Placed after login, before main app logic) ---
st.sidebar.markdown("---")
st.sidebar.subheader("模型选择")

available_models = ["--- 选择模型 ---", "Qwen/Qwen2.5-VL-7B-Instruct-AWQ", "OpenGVLab/InternVL3-8B"]
# Find the index of the currently selected model, default to 0 if none selected
current_index = 0
if st.session_state.selected_model_name in available_models:
    current_index = available_models.index(st.session_state.selected_model_name)

selected_model = st.sidebar.selectbox(
    "选择要使用的 LLM 模型:",
    available_models,
    index=current_index,
    key="model_selector" # Give it a key
)

# Display current status or load button
if st.session_state.model is not None and st.session_state.selected_model_name == selected_model:
    st.sidebar.success(f"模型 '{st.session_state.selected_model_name}' 已加载。")
elif selected_model != "--- 选择模型 ---":
    load_model_button = st.sidebar.button("加载模型", key="load_model_btn")
    if load_model_button:
        # Clear previous model from memory if exists and selection changed
        if st.session_state.model is not None:
            st.sidebar.info(f"正在卸载旧模型: {st.session_state.selected_model_name}...")
            del st.session_state.model
            del st.session_state.processor_instance
            st.session_state.model = None
            st.session_state.processor_instance = None
            st.session_state.selected_model_name = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            st.sidebar.info("旧模型已卸载。")


        st.sidebar.info(f"开始加载模型: {selected_model}...")
        with st.spinner(f"正在加载模型 {selected_model}..."):
            model, processor_instance = model_loader.load_model_and_processor(selected_model)
            if model and processor_instance:
                st.session_state.model = model
                st.session_state.processor_instance = processor_instance
                st.session_state.selected_model_name = selected_model
                st.sidebar.success(f"模型 {selected_model} 加载成功！")
                # Use st.rerun() cautiously, might clear other states unintentionally
                # Consider just letting the UI update naturally or using st.experimental_rerun() if needed
                st.rerun() # Rerun to update UI status
            else:
                st.sidebar.error(f"加载模型 {selected_model} 失败。")
                # Reset session state if loading failed
                st.session_state.model = None
                st.session_state.processor_instance = None
                st.session_state.selected_model_name = None
else:
    st.sidebar.warning("请选择一个模型并点击加载。")


# --- Main Application UI ---
st.title("📄 本地 LLM 文档处理应用")

# Retrieve model and processor from session state
model = st.session_state.get('model')
processor_instance = st.session_state.get('processor_instance')

# --- Sidebar (Continued) ---
with st.sidebar:
    st.header("操作面板")
    auth.add_logout_button() # Add logout button
    st.markdown("---")

    # 1. File Upload
    processed_images = file_handler.handle_file_upload()

    # AI智能推荐模板按钮 (Ensure model is loaded before enabling)
    if processed_images:
        st.markdown("---")
        st.subheader("AI 智能分析")
        ai_recommend = st.button(
            "🔍 AI 分析图片并推荐模板",
            key="ai_recommend_button",
            disabled=(model is None or processor_instance is None) # Disable if model not loaded
        )

        # 处理AI推荐请求
        if ai_recommend and processed_images and model and processor_instance:
            first_image = processed_images[0]
            preprocessor.handle_ai_recommendation(first_image, model, processor_instance)
        elif ai_recommend and (model is None or processor_instance is None):
             st.warning("请先加载模型以使用AI推荐功能。")
        elif ai_recommend:
            st.warning("请先上传图片以使用AI推荐功能。")

    # 2. Template Selection/Editing
    active_template = preprocessor.handle_preprocessing()

    # 3. Start Processing Button (Disable if model not loaded)
    st.markdown("---")
    start_processing = st.button(
        "🚀 开始处理",
        disabled=(not processed_images or not active_template or model is None or processor_instance is None) # Updated condition
    )

# --- Main Content Area ---
if start_processing:
    # Check again if model is loaded before processing
    if processed_images and active_template and model and processor_instance:
        with st.spinner("🧠 正在调用大模型处理图片..."):
            input_images_dict = {img_data['name']: img_data['image'] for img_data in processed_images}
            st.session_state.input_images = input_images_dict

            results = processor.process_images(
                images=processed_images,
                template=active_template,
                model=model, # Pass loaded model from session state
                processor=processor_instance # Pass loaded processor from session state
            )
            st.session_state.processing_results = results

            formatted_data = formatter.format_data_for_export(results, active_template)
            st.session_state.formatted_data = formatted_data
            st.session_state.edited_data = {} # Initialize/clear edited data
    elif not processed_images:
        st.warning("请先上传图片文件。")
    elif not active_template:
        st.warning("请选择或创建一个有效的处理模板。")
    elif model is None or processor_instance is None:
         st.warning("请先选择并成功加载一个模型。") # More specific warning

# Display results if they exist
if 'processing_results' in st.session_state:
    formatter.display_results(st.session_state.processing_results) # Step 3

# Add editable data section if formatted data exists
if 'formatted_data' in st.session_state and st.session_state.formatted_data:
    st.subheader("4. 核查与修改提取结果")
    st.info("您可以在下方的表格中直接修改提取的数据。修改后的数据将用于最终下载。点击“查看原图”可展开对应图片。")

    if 'edited_data' not in st.session_state:
        st.session_state.edited_data = {}

    input_images_dict = st.session_state.get('input_images', {})

    for filename, data in st.session_state.formatted_data.items():
        st.markdown(f"---")
        st.markdown(f"**文件: {filename}**")

        image_data = input_images_dict.get(filename)

        if image_data:
            with st.expander("查看原图", expanded=False):
                st.image(image_data, caption=f"原图: {filename}", use_container_width=True)
        else:
            st.caption("未找到对应的预览图。")

        if isinstance(data, pd.DataFrame) and not data.empty:
            # Ensure data is string type before editing for consistency
            data_str = data.astype(str)
            edited_df = st.data_editor(
                data_str, # Edit the string version
                key=f"editor_{filename}",
                num_rows="dynamic"
            )
            st.session_state.edited_data[filename] = edited_df # Store edited (string) data
        elif isinstance(data, pd.DataFrame) and data.empty:
            st.write("此文件未提取到表格数据。")
            st.session_state.edited_data[filename] = data # Store empty dataframe
        elif isinstance(data, str) and data.startswith("错误:"):
             st.error(f"处理错误，无法编辑: {data}")
             st.session_state.edited_data[filename] = data # Store error string
        elif isinstance(data, dict):
             st.write("数据为JSON对象，暂不支持直接编辑。")
             st.json(data)
             st.session_state.edited_data[filename] = data # Store dict
        else:
             st.write("数据格式未知或无法编辑。")
             # Ensure data is string before displaying/storing
             data_str = str(data)
             st.text(data_str)
             st.session_state.edited_data[filename] = data_str # Store as string


# Provide download buttons using the potentially edited data
if 'edited_data' in st.session_state and st.session_state.edited_data:
    formatter.provide_download_buttons(st.session_state.edited_data) # Step 5
elif 'formatted_data' in st.session_state:
     st.info("编辑表格后将启用下载功能。")
else:
    st.info("处理完成后，将在此处显示结果预览、编辑和下载选项。")

# Optional: Display uploaded image previews
if processed_images:
     with st.expander("查看已上传图片"):
         for img_data in processed_images:
             st.write(f"- {img_data['name']}")

# 添加对临时模板的清理逻辑
import atexit
atexit.register(helpers.cleanup_temp_templates)
