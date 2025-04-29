import streamlit as st
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
    elif not processed_images:
        st.warning("请先上传图片文件。")
    elif not active_template:
        st.warning("请选择或创建一个有效的处理模板。")

# Display results if they exist in session state
if 'processing_results' in st.session_state:
    formatter.display_results(st.session_state.processing_results)

# Provide download buttons if formatted data exists in session state
if 'formatted_data' in st.session_state:
    formatter.provide_download_buttons(st.session_state.formatted_data)
else:
    # Show placeholder if no processing has been done yet
    st.info("处理完成后，将在此处显示结果预览和下载选项。")

# Optional: Display uploaded image previews (can be simple names or thumbnails)
if processed_images:
     with st.expander("查看已上传图片"):
         for img_data in processed_images:
             st.write(f"- {img_data['name']}")
             # Optionally display small thumbnails
             # st.image(img_data['image'], width=100)
