import streamlit as st
import torch
# Import necessary classes from transformers
from transformers import AutoModelForCausalLM, AutoProcessor
# Import specific Qwen class and Auto classes from modelscope
from modelscope import Qwen2_5_VLForConditionalGeneration, AutoModel as MsAutoModel, AutoTokenizer as MsAutoTokenizer
import gc
from huggingface_hub.utils import RepositoryNotFoundError

# Removed @st.cache_resource
def load_model_and_processor(model_name: str):
    """Loads the specified VL model and processor/tokenizer, prioritizing ModelScope."""
    print(f"Attempting to load model and processor/tokenizer for: {model_name}")
    model = None
    processor = None # This will hold processor for Qwen, tokenizer for InternVL
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_loaded_from = None # Track source
    processor_loaded_from = None # Track source

    try:
        if model_name == "Qwen/Qwen2.5-VL-7B-Instruct-AWQ":
            # --- Try loading Qwen from ModelScope first (using specific class) ---
            print(f"Attempting to load Qwen model '{model_name}' from ModelScope (Specific Class)...")
            try:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_name,
                    attn_implementation="flash_attention_2",
                    device_map=device,
                )
                model_loaded_from = "ModelScope (Specific Class)"
                print(f"Qwen model loaded successfully from {model_loaded_from}.")
            except Exception as e_ms:
                print(f"Failed to load Qwen from ModelScope using specific class ({e_ms}). Trying generic AutoModel (Transformers)...")
                # --- Fallback to generic AutoModel (Transformers, might try HF) ---
                try:
                    print(f"Attempting to load Qwen model '{model_name}' using AutoModelForCausalLM (Transformers)...")
                    # Explicitly use Transformers AutoModel here for fallback
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                        low_cpu_mem_usage=True,
                        trust_remote_code=True,
                        device_map=device
                    ).eval()
                    model_loaded_from = "Transformers AutoModelForCausalLM (Fallback)"
                    print(f"Qwen model loaded successfully using {model_loaded_from}.")
                except Exception as e_hf:
                    print(f"Failed to load Qwen model '{model_name}' using Transformers AutoModelForCausalLM: {e_hf}")
                    raise e_hf # Re-raise the error if fallback also fails

            # --- Load Qwen Processor (Use Transformers AutoProcessor, likely tries HF/MS) ---
            # AutoProcessor from Transformers often handles both hubs reasonably well.
            print(f"Attempting to load processor for '{model_name}' using AutoProcessor (Transformers)...")
            min_pixels = 1024 * 28 * 28
            max_pixels = 16384 * 28 * 28
            try:
                processor = AutoProcessor.from_pretrained(
                    model_name,
                    use_fast=True,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
                    vl_high_resolution_images=True,
                    trust_remote_code=True
                )
                processor_loaded_from = "Transformers AutoProcessor"
                print(f"Qwen processor loaded successfully using {processor_loaded_from}.")
            except Exception as e_proc:
                 print(f"Failed to load Qwen processor for '{model_name}' using AutoProcessor: {e_proc}")
                 raise e_proc # Re-raise error

        elif model_name in ["OpenGVLab/InternVL3-8B", "OpenGVLab/InternVL3-2B"]:
            # --- Load InternVL Model (Explicitly use ModelScope AutoModel) ---
            print(f"Attempting to load InternVL model '{model_name}' using ModelScope AutoModel...")
            try:
                # Use ModelScope AutoModel explicitly
                model = MsAutoModel.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    # use_flash_attn=True, # Optional
                    device_map=device
                ).eval()
                model_loaded_from = "ModelScope AutoModel"
                print(f"InternVL model loaded successfully using {model_loaded_from}.")
            except Exception as e_ms:
                print(f"Failed to load InternVL model '{model_name}' using ModelScope AutoModel: {e_ms}.")
                # Optional: Add fallback to Transformers AutoModel if desired
                # print(f"Attempting fallback using Transformers AutoModel...")
                # try:
                #     from transformers import AutoModel as TfAutoModel
                #     model = TfAutoModel.from_pretrained(...)
                #     model_loaded_from = "Transformers AutoModel (Fallback)"
                #     print(...)
                # except Exception as e_tf:
                #     print(...)
                #     raise e_ms # Re-raise original error if fallback also fails
                raise e_ms # Re-raise error if loading fails

            # --- Load InternVL Tokenizer (Explicitly use ModelScope AutoTokenizer) ---
            print(f"Attempting to load tokenizer for '{model_name}' using ModelScope AutoTokenizer...")
            try:
                 # Use ModelScope AutoTokenizer explicitly
                 processor = MsAutoTokenizer.from_pretrained(
                     model_name,
                     trust_remote_code=True,
                     use_fast=False
                 )
                 processor_loaded_from = "ModelScope AutoTokenizer"
                 print(f"InternVL tokenizer loaded successfully using {processor_loaded_from}.")
            except Exception as e_tok:
                 print(f"Failed to load InternVL tokenizer using ModelScope AutoTokenizer: {e_tok}")
                 # Optional: Add fallback to Transformers AutoTokenizer if desired
                 # print(f"Attempting fallback using Transformers AutoTokenizer...")
                 # try:
                 #     from transformers import AutoTokenizer as TfAutoTokenizer
                 #     processor = TfAutoTokenizer.from_pretrained(...)
                 #     processor_loaded_from = "Transformers AutoTokenizer (Fallback)"
                 #     print(...)
                 # except Exception as e_tf_tok:
                 #     print(...)
                 #     raise e_tok # Re-raise original error
                 raise e_tok # Re-raise error

        else:
            st.error(f"Unsupported model selected: {model_name}")
            return None, None

        # Final check
        if model is None or processor is None:
             raise ValueError(f"Model or processor/tokenizer failed to load for {model_name}")

        print(f"Final check: Model loaded from: {model_loaded_from}, Processor/Tokenizer loaded from: {processor_loaded_from}")
        return model, processor # Return model and processor/tokenizer

    except ImportError as e:
        if "flash_attn" in str(e):
             st.error("Error: flash_attn package not found. Please install it: pip install flash-attn --no-build-isolation")
        else:
             st.error(f"Import Error: {e}. Please ensure all dependencies are installed.")
        # Clean up memory
        del model
        del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
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
