import streamlit as st
from PIL import Image
import io

def handle_file_upload():
    """Handles single or multiple file uploads in the Streamlit sidebar."""
    uploaded_files = st.sidebar.file_uploader(
        "上传图片文件 (可多选)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True # Allow multiple files
    )

    processed_images = []
    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                # Read file content as bytes
                bytes_data = uploaded_file.getvalue()
                # Open image from bytes
                raw_image = Image.open(io.BytesIO(bytes_data)).convert('RGB')
                processed_images.append({"name": uploaded_file.name, "image": raw_image})
            except Exception as e:
                st.sidebar.error(f"无法加载文件 {uploaded_file.name}: {e}")
    else:
        st.sidebar.info("请上传至少一张图片文件。")

    return processed_images
