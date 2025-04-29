import streamlit as st
import torch
from typing import List, Dict, Any
import gc # Import garbage collector

def generate_prompt_from_template(template: Dict[str, Any]) -> str:
    """Generates an LLM prompt based on the provided template."""
    if not template or 'fields' not in template:
        return "请提供图片内容描述。" # Default basic prompt

    field_details = []
    for field in template.get('fields', []):
        detail = f"- {field.get('name', '未知字段')}"
        if field.get('format'):
            detail += f" (格式提示: {field.get('format')})"
        if field.get('required', False):
            detail += " (必须提取)"
        field_details.append(detail)

    fields_string = "\n".join(field_details)
    output_hint = template.get('output_format_hint', '文本')
    notes = template.get('notes', '')

    prompt = f"""你是一个专业的图像数据处理助手。
请分析提供的图片，并严格按照以下要求提取信息：

**需要提取的字段:**
{fields_string}

**输出格式要求:**
请尽量将提取的结果组织成 **{output_hint}** 格式。
如果输出格式是 CSV，请只输出数据行，不要包含表头。
如果无法提取某个必须字段，请在该字段位置填写 "NULL" 或留空。

"""
    if notes:
        prompt += f"""
**补充说明:**
{notes}

"""
    prompt += "请开始处理图片。"
    return prompt

def process_images(
    images: List[Dict[str, Any]], 
    template: Dict[str, Any],
    model,
    processor
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
        st.error("模型或处理器未加载。")
        return results

    st.info(f"准备处理 {len(images)} 张图片...")
    progress_bar = st.progress(0)

    for i, img_data in enumerate(images):
        image_name = img_data['name']
        raw_image = img_data['image']
        st.write(f"正在处理: {image_name}")

        # 1. Generate prompt for this image based on the template
        prompt = generate_prompt_from_template(template)

        # 2. Prepare input for the model
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image"}, # Placeholder for the image
                ],
            }
        ]
        try:
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(
                text=[text],
                images=[raw_image],
                padding=True,
                return_tensors="pt",
            ).to(model.device) # Use model's device

            # 3. Generate response
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=1536, # Increased token limit slightly
                    do_sample=False
                )

            # 4. Decode the output
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            # 获取结果后立即保存到变量中
            result_text = output_text[0] if output_text else ""
            
            # 然后再清理内存
            del inputs, generated_ids, generated_ids_trimmed, output_text, text
            # Manually trigger garbage collection
            gc.collect()
            # Clear CUDA cache to free GPU memory if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            results.append({"name": image_name, "raw_output": result_text, "prompt": prompt})
            st.success(f"处理完成: {image_name}")

        except Exception as e:
            error_msg = f"处理图片 '{image_name}' 时出错: {e}"
            st.error(error_msg)
            results.append({"name": image_name, "raw_output": f"错误: {error_msg}", "prompt": prompt, "error": True})

        # Update progress bar
        progress_bar.progress((i + 1) / len(images))

    st.info("所有图片处理完毕。")
    return results
