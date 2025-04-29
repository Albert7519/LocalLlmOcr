import streamlit as st
import torch
from modelscope import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
import io # To handle byte stream from file uploader

# --- Cached Model Loading ---
# Use st.cache_resource to load the model and processor only once
@st.cache_resource
def load_model_and_processor():
    """Loads the Qwen VL model and processor, cached by Streamlit.""" # Reformat docstring line
    print("Loading model and processor (this should run only once)...")
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2", # Requires flash-attn installation
            device_map="cuda", # Load directly to GPU
        )
        # --- Add min_pixels and max_pixels ---
        min_pixels = 1024 * 28 * 28 # Encourage higher detail processing
        max_pixels = 16384 * 28 * 28 # Set an upper limit (adjust as needed)
        processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
            use_fast=True,
            min_pixels=min_pixels, # Add min_pixels
            max_pixels=max_pixels,  # Add max_pixels
            vl_high_resolution_images=True # Add high resolution flag
        )
        print(f"Model and processor loaded successfully (min_pixels={min_pixels}, max_pixels={max_pixels}, vl_high_resolution_images=True).")
        return model, processor
    except ImportError:
        st.error("Error: flash_attn package not found. Please install it: pip install flash-attn --no-build-isolation")
        return None, None
    except Exception as e:
        st.error(f"Error loading model or processor: {e}")
        return None, None

# --- Streamlit UI Setup ---
st.set_page_config(layout="wide") # Use wide layout for better display
st.title("🧾 Qwen-VL 发票 OCR 应用")

# Load model and processor using the cached function
model, processor = load_model_and_processor()

# Sidebar for inputs
with st.sidebar:
    st.header("输入设置")
    # File uploader for the receipt image
    uploaded_file = st.file_uploader("上传发票图片", type=["jpg", "jpeg", "png"])

    # Text area for the prompt
    default_prompt = """你是一个专业的图像数据处理助手，负责从提供的发票图片中提取信息，并将其转换为标准化的CSV表格格式。

请分析以下图片，并执行以下任务：

1.  **识别关键信息**：从图片中找出以下发票信息：
    *   发票日期
    *   金额
    *   发票号码
    *   车号
    *   上车时间
    *   下车时间

2.  **规范化输出**：将提取的信息整理成CSV格式，字段顺序和格式要求如下：
    *   发票日期 (格式: YYYY-MM-DD)
    *   金额 (单位: 元，保留两位小数)
    *   发票号码
    *   车号 (格式: XX-XXXXXX)
    *   上车时间 (格式: HH:MM)
    *   下车时间 (格式: HH:MM)

3.  **处理缺失值**：
    *   如果图片中找不到发票日期，请在CSV对应位置填写 "NULL"。
    *   如果图片中找不到金额，请在CSV对应位置填写 "NULL"。
    *   如果图片中找不到发票号码，请在CSV对应位置填写 "NULL"。
    *   如果图片中找不到车号，请在CSV对应位置填写 "NULL"。
    *   如果图片中找不到上车时间，请在CSV对应位置填写 "NULL"。
    *   如果图片中找不到下车时间，请在CSV对应位置填写 "NULL"。

4.  **输出格式示例**：
    ```csv
    发票日期,金额,发票号码,车号,上车时间,下车时间
    YYYY-MM-DD,XX.XX,XXXXXXXX,XX-XXXXXX,HH:MM,HH:MM
    ```
    请只输出CSV内容，不要包含表头。

5. **注意事项**：
    如果存在税前价格和税后价格，只统计税后价格。

请根据以上规则处理图片中的发票信息。"""

    prompt = st.text_area("提取指令 (Prompt)", value=default_prompt, height=150)

    # Button to trigger OCR
    submit_button = st.button("开始提取信息")

# Main area for displaying image and results
col1, col2 = st.columns(2)

with col1:
    st.subheader("上传的图片")
    if uploaded_file is not None:
        # Display the uploaded image
        try:
            raw_image = Image.open(uploaded_file).convert('RGB')
            st.image(raw_image, caption="Uploaded Receipt", use_container_width=True)
        except Exception as e:
            st.error(f"无法加载图片: {e}")
            raw_image = None # Ensure raw_image is None if loading fails
    else:
        st.info("请在左侧上传一张发票图片。")
        raw_image = None

with col2:
    st.subheader("提取结果")
    # Placeholder for the results
    result_placeholder = st.empty()
    result_placeholder.info("点击“开始提取信息”按钮后，结果将显示在这里。")

# --- OCR Processing Logic ---
if submit_button and raw_image is not None and model is not None and processor is not None:
    with st.spinner("正在处理图片并提取信息..."):
        try:
            # Structure the input messages (similar to ocr_receipt.py)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image"}, # Placeholder for the image
                    ],
                }
            ]

            # Prepare inputs using the processor
            # Note: We pass the raw PIL image directly
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(
                text=[text],
                images=[raw_image], # Pass the loaded PIL Image object
                padding=True,
                return_tensors="pt",
            ).to("cuda")

            # Generate response
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=False # Use greedy decoding
                )

            # Decode the output
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            # Display the result
            if output_text:
                result_placeholder.markdown(output_text[0]) # Use markdown for better formatting if needed
            else:
                result_placeholder.warning("模型没有生成任何文本。")

        except Exception as e:
            result_placeholder.error(f"处理过程中发生错误: {e}")
elif submit_button and raw_image is None:
    st.sidebar.warning("请先上传一张图片。")
elif submit_button and (model is None or processor is None):
     st.sidebar.error("模型或处理器未能成功加载，无法进行处理。")
