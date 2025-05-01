import streamlit as st
import torch
from typing import List, Dict, Any, Optional
import gc
from PIL import Image # Add PIL
import torchvision.transforms as T # Add torchvision
from torchvision.transforms.functional import InterpolationMode # Add InterpolationMode
import numpy as np # Add numpy
import math # Add math
import traceback # Import traceback module

# --- Add Image Preprocessing Helpers for InternVL (from example) ---
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            # Calculate area based on potential resizing to image_size blocks
            current_area = best_ratio[0] * best_ratio[1] * (image_size**2)
            new_area = ratio[0] * ratio[1] * (image_size**2)
            # Prefer the ratio that results in a larger area after potential resizing
            # This interpretation might differ slightly from original, aims for clarity
            if new_area > current_area:
                 best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    # Assuming image is a PIL Image object
    orig_width, orig_height = image.size
    if orig_height == 0 or orig_width == 0:
        print("Warning: Original image dimensions are zero.")
        return [] # Return empty list if image dimensions are invalid

    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # Handle cases where target_ratios might be empty
    if not target_ratios:
        target_ratios = {(1, 1)} # Default to 1x1 if range is invalid

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # Calculate the target width and height based on selected ratio and image_size
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # Ensure target dimensions are integers and non-zero
    target_width = max(1, int(round(target_width)))
    target_height = max(1, int(round(target_height)))

    resized_img = image.resize((target_width, target_height))
    processed_images = []

    # Calculate how many patches fit horizontally and vertically
    patches_w = target_width // image_size
    patches_h = target_height // image_size
    actual_blocks = patches_w * patches_h

    if actual_blocks == 0: # Handle cases where image is smaller than patch size
        # Resize the whole image to the patch size
        split_img = resized_img.resize((image_size, image_size))
        processed_images.append(split_img)
        actual_blocks = 1 # We have one block now
    else:
        for i in range(actual_blocks):
            row = i // patches_w
            col = i % patches_w
            box = (
                col * image_size,
                row * image_size,
                (col + 1) * image_size,
                (row + 1) * image_size
            )
            split_img = resized_img.crop(box)
            # Ensure the cropped image is exactly image_size x image_size
            if split_img.size != (image_size, image_size):
                split_img = split_img.resize((image_size, image_size))
            processed_images.append(split_img)

    # The assertion might be too strict if rounding/integer division causes mismatches
    # assert len(processed_images) == blocks, f"Expected {blocks} blocks, got {len(processed_images)}"

    if use_thumbnail and actual_blocks != 1: # Add thumbnail if more than one patch
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)

    return processed_images


def load_image_internvl(image, input_size=448, max_num=12):
    # Assuming image is a PIL Image object
    transform = build_transform(input_size=input_size)
    # Use thumbnail=True as per the single image example logic implicitly
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    if not images: # Handle case where dynamic_preprocess returns empty
        print("Warning: dynamic_preprocess returned no images.")
        # Optionally return a dummy tensor or raise an error
        # Returning a single black image tensor as a fallback
        dummy_tensor = torch.zeros(3, input_size, input_size)
        return torch.stack([dummy_tensor])

    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# --- End Image Preprocessing Helpers ---

# --- Define Prompt Generation Function ---
def generate_prompt_from_template(template: Dict[str, Any], custom_prompt: Optional[str] = None) -> str:
    """Generates a detailed prompt for the LLM based on the template."""
    if custom_prompt:
        # If a custom prompt is provided, use it directly.
        # This is useful for tasks like AI template recommendation.
        return custom_prompt

    # Construct the prompt based on the template fields
    prompt_parts = []
    prompt_parts.append(f"你是一个专业的文档分析和信息提取助手。请仔细分析提供的图片内容，并根据以下要求提取信息：")
    prompt_parts.append(f"\n文档类型描述: {template.get('description', '未指定')}")

    prompt_parts.append("\n需要提取的字段:")
    fields = template.get('fields', [])
    if not fields:
        prompt_parts.append("- (未指定具体字段，请尝试提取所有关键信息)")
    else:
        for i, field in enumerate(fields):
            field_name = field.get('name', f'字段{i+1}')
            field_format = field.get('format', '无特定格式')
            field_required = "必须提取" if field.get('required', False) else "可选提取"
            prompt_parts.append(f"- {field_name}: (格式: {field_format}, 要求: {field_required})")

    output_hint = template.get('output_format_hint', 'CSV')
    prompt_parts.append(f"\n输出格式要求: 请严格按照 {output_hint} 格式组织提取结果。")
    if output_hint == 'CSV':
        prompt_parts.append("  - CSV格式要求: 使用逗号分隔值，第一行为表头（字段名称），后续行为对应的数据。如果某个字段未找到，请留空或填写'NULL'。")
    elif output_hint == 'JSON':
        prompt_parts.append("  - JSON格式要求: 输出一个包含提取结果的JSON对象或JSON对象列表。键为字段名称，值为提取的数据。")
    elif output_hint == 'XLSX':
         prompt_parts.append("  - XLSX格式要求: 结果应能方便地转为Excel表格，类似CSV格式，第一行为表头，后续行为数据。")


    notes = template.get('notes', '')
    if notes:
        prompt_parts.append(f"\n补充说明:\n{notes}")

    prompt_parts.append("\n请开始分析图片并按要求提取信息。")

    return "\n".join(prompt_parts)
# --- End Prompt Generation Function ---


def process_images(
    images: List[Dict[str, Any]],
    template: Dict[str, Any],
    model,
    processor, # This will be AutoProcessor for Qwen, AutoTokenizer for InternVL
    custom_prompt: Optional[str] = None,
    single_image_mode: bool = False
) -> List[Dict[str, Any]]:
    """Processes a list of images using the LLM based on the template."""
    results = []
    if not images:
        st.warning("没有需要处理的图片。")
        return results
    if not template:
        st.error("没有提供有效的处理模板。")
        return results
    if not model or not processor:
        st.error("模型或处理器/Tokenizer未加载。")
        return results

    # 如果是single_image_mode，不显示进度信息，用于内部调用
    if not single_image_mode:
        st.info(f"准备处理 {len(images)} 张图片...")
        progress_bar = st.progress(0)

    for i, img_data in enumerate(images):
        image_name = img_data['name']
        raw_image = img_data['image'] # This is a PIL Image

        if not single_image_mode:
            st.write(f"正在处理: {image_name}")

        # 1. Generate prompt for this image based on the template
        # This call should now work as the function is defined above
        prompt = generate_prompt_from_template(template, custom_prompt)

        # 2. Prepare input for the model - Conditional based on model type
        selected_model_name = st.session_state.get('selected_model_name', '')
        inputs = None # For Qwen, holds processor output dict
        pixel_values = None # Holds image tensors
        input_ids = None # Holds text token ids
        attention_mask = None # Holds attention mask

        try:
            if "Qwen" in selected_model_name:
                # --- Qwen-VL Input Preparation (using AutoProcessor) ---
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image"}, # Placeholder for the image
                        ],
                    }
                ]
                # 'processor' is the AutoProcessor instance here
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(
                    text=[text],
                    images=[raw_image], # Pass PIL image directly
                    padding=True,
                    return_tensors="pt",
                ).to(model.device)
                # Keep input_ids for calculating token length later
                input_ids = inputs.get('input_ids')
                # --- End Qwen-VL Input ---

                # 3. Generate response (Qwen)
                with torch.no_grad():
                    gen_kwargs = {
                        "max_new_tokens": 1536,
                        "do_sample": False, # Use greedy decoding for consistency
                    }
                    # Pass the entire 'inputs' dictionary unpacked
                    generated_ids = model.generate(
                        **inputs, # Pass all prepared inputs
                        **gen_kwargs
                    )
                # 4. Decode the output (Qwen)
                current_processor_or_tokenizer = processor
                input_token_len = input_ids.shape[1] if input_ids is not None else 0
                if generated_ids.shape[1] > input_token_len:
                     generated_ids_trimmed = generated_ids[:, input_token_len:]
                else:
                     generated_ids_trimmed = torch.tensor([[]], dtype=torch.long, device=generated_ids.device)
                output_text = current_processor_or_tokenizer.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                result_text = output_text[0] if output_text else ""


            elif "InternVL" in selected_model_name:
                # --- InternVL Input Preparation (using AutoTokenizer + custom preprocessing) ---
                # Here, 'processor' is the AutoTokenizer instance
                tokenizer = processor
                # Prepend '<image>\\n' as per example structure
                internvl_prompt = f"<image>\\n{prompt}"

                # Tokenize the text prompt
                tokenized_inputs = tokenizer(internvl_prompt, return_tensors="pt")
                input_ids = tokenized_inputs.input_ids.to(model.device)
                attention_mask = tokenized_inputs.attention_mask.to(model.device)

                # Preprocess the image using the helper functions
                image_input_size = 448 # Default from example, make configurable later if needed
                image_max_num = 12    # Default from example
                pixel_values = load_image_internvl(
                    raw_image, # Pass PIL image
                    input_size=image_input_size,
                    max_num=image_max_num
                ).to(model.dtype).to(model.device) # Ensure dtype and device match model

                # 3. Generate response (InternVL using model.chat)
                with torch.no_grad():
                    # Define generation config for model.chat
                    generation_config = dict(
                        max_new_tokens=1536,
                        do_sample=False # Use greedy decoding for consistency
                        # Add other relevant generation parameters if needed
                    )
                    # Use model.chat for InternVL
                    # model.chat expects tokenizer, pixel_values, question, generation_config
                    # It typically returns the response text directly, or (response, history)
                    response = model.chat(
                        tokenizer=tokenizer, # Pass the tokenizer instance
                        pixel_values=pixel_values, # Pass the preprocessed image tensor
                        question=internvl_prompt, # Pass the combined prompt
                        generation_config=generation_config,
                        history=None, # No history for single turn
                        return_history=False # We only need the response text
                    )
                    result_text = response # Assign the response directly

                # 4. Decode the output (InternVL - already decoded by model.chat)
                # No separate decoding step needed as model.chat returns text

            else:
                # Fallback or error for unsupported models
                st.error(f"Unsupported model type for input preparation: {selected_model_name}")
                results.append({"name": image_name, "raw_output": f"错误: 不支持的模型类型 {selected_model_name}", "prompt": prompt, "error": True})
                continue # Skip to next image

            # 获取结果后立即保存到变量中
            result_text = result_text if 'result_text' in locals() else ""

            # 然后再清理内存
            # Use locals() to check for variable existence before deleting
            vars_to_delete = ['inputs', 'generated_ids', 'generated_ids_trimmed', 'output_text', 'pixel_values', 'input_ids', 'attention_mask', 'text', 'tokenized_inputs']
            for var_name in vars_to_delete:
                if var_name in locals():
                    try:
                        # print(f"Deleting {var_name}") # Debug print
                        del locals()[var_name]
                    except NameError:
                        pass # Should not happen with locals() check

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            results.append({"name": image_name, "raw_output": result_text, "prompt": prompt}) # Store original prompt

            if not single_image_mode:
                st.success(f"处理完成: {image_name}")

        except Exception as e:
            # Capture full traceback
            tb_str = traceback.format_exc()
            # Include traceback in the error message
            error_msg = f"处理图片 '{image_name}' 时出错: {e}\n\nTraceback:\n{tb_str}"
            if not single_image_mode:
                st.error(error_msg) # Display full error in UI
            results.append({"name": image_name, "raw_output": f"错误: {error_msg}", "prompt": prompt, "error": True}) # Store full error
            # Ensure cleanup even on error
            vars_to_delete = ['inputs', 'generated_ids', 'generated_ids_trimmed', 'output_text', 'pixel_values', 'input_ids', 'attention_mask', 'text', 'tokenized_inputs']
            for var_name in vars_to_delete:
                if var_name in locals():
                    try:
                        # print(f"Deleting {var_name} (in except)") # Debug print
                        del locals()[var_name]
                    except NameError:
                        pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Update progress bar if not in single image mode
        if not single_image_mode and 'progress_bar' in locals():
            progress_bar.progress((i + 1) / len(images))

    if not single_image_mode:
        st.info("所有图片处理完毕。")
    return results
