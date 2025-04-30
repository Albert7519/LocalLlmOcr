import streamlit as st
import torch
# Use AutoModelForCausalLM and AutoTokenizer/AutoProcessor for broader compatibility
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from modelscope import Qwen2_5_VLForConditionalGeneration # Keep specific class if needed
import gc

# Removed @st.cache_resource
def load_model_and_processor(model_name: str):
    """Loads the specified VL model and processor."""
    print(f"Attempting to load model and processor for: {model_name}")
    try:
        model = None
        processor = None
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if model_name == "Qwen/Qwen2.5-VL-7B-Instruct-AWQ":
            # Keep existing Qwen loading logic
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                # torch_dtype=torch.float16, # AWQ handles dtype
                attn_implementation="flash_attention_2",
                device_map=device,
            )
            # --- Add min_pixels and max_pixels ---
            min_pixels = 1024 * 28 * 28 # Encourage higher detail processing
            max_pixels = 16384 * 28 * 28 # Set an upper limit (adjust as needed)
            processor = AutoProcessor.from_pretrained(
                model_name,
                use_fast=True,
                min_pixels=min_pixels, # Add min_pixels
                max_pixels=max_pixels,  # Add max_pixels
                vl_high_resolution_images=True # Add high resolution flag
            )
            print(f"Qwen model and processor loaded successfully (min_pixels={min_pixels}, max_pixels={max_pixels}, vl_high_resolution_images=True).")

        elif model_name == "OpenGVLab/InternVL3-8B":
            # Add loading logic for InternVL3-8B using transformers
            # Check ModelScope Hub for precise loading instructions if needed
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16, # Use bfloat16 if available
                low_cpu_mem_usage=True,
                trust_remote_code=True, # Often required for custom model code
                device_map=device
            ).eval() # Set to evaluation mode

            # InternVL might use AutoTokenizer as its processor or have a specific AutoProcessor
            # Let's try AutoProcessor first, fallback to AutoTokenizer if needed
            try:
                 processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
                 print("InternVL processor loaded using AutoProcessor.")
            except Exception as e_proc:
                 print(f"Failed to load InternVL processor using AutoProcessor ({e_proc}), trying AutoTokenizer...")
                 try:
                     processor = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                     print("InternVL processor loaded using AutoTokenizer.")
                 except Exception as e_tok:
                     st.error(f"Failed to load processor/tokenizer for InternVL: {e_tok}")
                     return None, None
            print("InternVL model and processor loaded successfully.")

        else:
            st.error(f"Unsupported model selected: {model_name}")
            return None, None

        return model, processor

    except ImportError as e:
        if "flash_attn" in str(e):
             st.error("Error: flash_attn package not found. Please install it: pip install flash-attn --no-build-isolation")
        else:
             st.error(f"Import Error: {e}. Please ensure all dependencies are installed.")
        return None, None
    except Exception as e:
        st.error(f"Error loading model or processor for {model_name}: {e}")
        # Clean up memory if loading failed partially
        del model
        del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None, None
