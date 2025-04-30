import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from modelscope import Qwen2_5_VLForConditionalGeneration
import gc

def load_model_and_processor(model_name: str):
    """Loads the specified VL model and processor."""
    print(f"Attempting to load model and processor for: {model_name}")
    try:
        model = None
        processor = None
        # --- Modified Device Logic ---
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available(): # Keep CUDA check just in case, though unlikely on Mac
            device = "cuda"
        else:
            device = "cpu"
        print(f"Using device: {device}")
        # --- End Modified Device Logic ---

        # Determine appropriate dtype (bfloat16 might not be fully supported on MPS, float16 is safer)
        compute_dtype = torch.float16

        if model_name == "Qwen/Qwen2.5-VL-7B-Instruct-AWQ":
            # --- Modified Qwen-7B-AWQ Loading ---
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                # attn_implementation="flash_attention_2", # REMOVED: flash-attn not available
                device_map=device, # Load directly to the determined device
                # AWQ might have specific dtype requirements, check model card if issues arise
            )
            # --- End Modified Qwen-7B-AWQ Loading ---

            # --- Processor loading (remains similar) ---
            min_pixels = 1024 * 28 * 28
            max_pixels = 16384 * 28 * 28
            processor = AutoProcessor.from_pretrained(
                model_name,
                use_fast=True,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                vl_high_resolution_images=True
            )
            print(f"Qwen 7B AWQ model and processor loaded successfully on {device}.")

        # --- Added Qwen 72B loading logic from previous step (ensure device_map is correct) ---
        elif model_name == "Qwen/Qwen2.5-VL-72B-Instruct":
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=compute_dtype, # Use float16 for MPS
                    device_map=device, # Load directly to the determined device
                    trust_remote_code=True,
                    low_cpu_mem_usage=True # Still useful even with MPS
                ).eval()

                min_pixels = 1024 * 28 * 28
                max_pixels = 16384 * 28 * 28
                processor = AutoProcessor.from_pretrained(
                    model_name,
                    use_fast=True,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
                    vl_high_resolution_images=True,
                    trust_remote_code=True
                )
                print(f"Qwen 72B model and processor loaded successfully on {device} (dtype={model.dtype}).")

            except ImportError as e:
                 # This block might not be needed if flash-attn is removed from requirements
                 # but kept for robustness if some internal code still tries to import it.
                 if "flash_attn" in str(e):
                     st.warning("flash_attn import failed (expected on Mac). Using default attention.")
                     # Attempt loading without flash_attn hint (already removed above)
                     model = AutoModelForCausalLM.from_pretrained(
                         model_name,
                         torch_dtype=compute_dtype,
                         device_map=device,
                         trust_remote_code=True,
                         low_cpu_mem_usage=True
                     ).eval()
                     processor = AutoProcessor.from_pretrained(
                         model_name, use_fast=True, min_pixels=min_pixels, max_pixels=max_pixels,
                         vl_high_resolution_images=True, trust_remote_code=True
                     )
                     print(f"Qwen 72B model (without flash_attn) and processor loaded successfully on {device} (dtype={model.dtype}).")
                 else:
                     raise e

        elif model_name == "OpenGVLab/InternVL3-8B":
            # --- Modified InternVL Loading ---
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=compute_dtype, # Use float16 for MPS
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                device_map=device # Load directly to the determined device
            ).eval()
            # --- End Modified InternVL Loading ---

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
            print(f"InternVL model and processor loaded successfully on {device}.")

        else:
            st.error(f"Unsupported model selected: {model_name}")
            return None, None

        return model, processor

    except ImportError as e:
         # Removed specific flash_attn error message here as it's handled elsewhere or expected
         st.error(f"Import Error: {e}. Please ensure all dependencies are installed correctly for your system (macOS).")
         return None, None
    except Exception as e:
        st.error(f"Error loading model or processor for {model_name} on {device}: {e}")
        # Clean up memory if loading failed partially
        # Use 'is not None' checks before deleting
        if 'model' in locals() and model is not None: del model
        if 'processor' in locals() and processor is not None: del processor
        gc.collect()
        # No torch.cuda.empty_cache() needed for MPS/CPU
        return None, None
