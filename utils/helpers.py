import os
import json
import streamlit as st
import uuid

TEMPLATES_DIR = "templates"

def load_template(template_name: str):
    """Loads a specific JSON template file."""
    filepath = os.path.join(TEMPLATES_DIR, f"{template_name}.json")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"模板文件未找到: {filepath}")
        return None
    except json.JSONDecodeError:
        st.error(f"模板文件格式错误: {filepath}")
        return None
    except Exception as e:
        st.error(f"加载模板时出错 {filepath}: {e}")
        return None

def list_available_templates():
    """Lists available template files (without .json extension)."""
    try:
        files = [f.replace(".json", "") for f in os.listdir(TEMPLATES_DIR) if f.endswith(".json")]
        return files
    except FileNotFoundError:
        st.error(f"模板目录 '{TEMPLATES_DIR}' 不存在.")
        return []
    except Exception as e:
        st.error(f"列出模板时出错: {e}")
        return []

def save_template(template_name: str, template_data: dict):
    """Saves template data to a JSON file."""
    filepath = os.path.join(TEMPLATES_DIR, f"{template_name}.json")
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(template_data, f, ensure_ascii=False, indent=2)
        st.success(f"模板 '{template_name}' 已保存.")
        return True
    except Exception as e:
        st.error(f"保存模板时出错 {filepath}: {e}")
        return False

def create_temp_template_id():
    """生成临时模板ID，用于智能推荐功能"""
    return f"temp_{uuid.uuid4().hex[:8]}"

def save_temp_template(template_data: dict) -> str:
    """保存临时模板，返回模板ID"""
    temp_id = create_temp_template_id()
    filepath = os.path.join(TEMPLATES_DIR, f"{temp_id}.json")
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(template_data, f, ensure_ascii=False, indent=2)
        return temp_id
    except Exception as e:
        st.error(f"保存临时模板时出错: {e}")
        return None

def cleanup_temp_templates():
    """清理temp_开头的临时模板文件"""
    try:
        for filename in os.listdir(TEMPLATES_DIR):
            if filename.startswith("temp_") and filename.endswith(".json"):
                os.remove(os.path.join(TEMPLATES_DIR, filename))
    except Exception as e:
        print(f"清理临时模板时出错: {e}")
