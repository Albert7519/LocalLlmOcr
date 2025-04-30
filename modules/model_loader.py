import streamlit as st
import torch
from modelscope import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# Use st.cache_resource to load the model and processor only once
@st.cache_resource
def load_model_and_processor():
    """Loads the Qwen VL model and processor, cached by Streamlit."""
    print("Loading model and processor (this should run only once)...")
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
            # torch_dtype=torch.float16, # Removed to let the library handle AWQ loading
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
