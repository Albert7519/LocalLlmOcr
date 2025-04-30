import streamlit as st
import torch # Add torch import
import gc # Add gc import

def show_login_form():
    """Displays a simple login form and returns True if login is successful (placeholder)."""
    # Initialize login status in session state if it doesn't exist
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    # If already logged in, don't show the form
    if st.session_state.logged_in:
        return True

    # Display login form
    st.title("登录")
    st.write("请输入凭据以访问应用 (此为占位符，任意输入即可)")

    with st.form("login_form"):
        username = st.text_input("用户名")
        password = st.text_input("密码", type="password")
        submitted = st.form_submit_button("登录")

        if submitted:
            if username and password:
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("请输入用户名和密码")

    return False

def add_logout_button():
    """Adds a logout button to the sidebar if logged in."""
    if st.session_state.get('logged_in', False):
        if st.sidebar.button("退出登录"):
            # Clear model from memory before clearing session state
            if 'model' in st.session_state and st.session_state.model is not None:
                 print("Logging out: Clearing model from memory...")
                 del st.session_state.model
                 del st.session_state.processor_instance
                 gc.collect()
                 if torch.cuda.is_available():
                     torch.cuda.empty_cache()
                 print("Model cleared.")

            st.session_state.logged_in = False
            # Clear other relevant session state
            keys_to_clear = [
                'processing_results', 'formatted_data', 'edited_data',
                'selected_template_option', 'manual_template',
                'loaded_template', 'loaded_template_name', 'edited_loaded_template',
                'input_images',
                # Add model related keys
                'selected_model_name', 'model', 'processor_instance'
            ]
            for key in keys_to_clear:
                if key in st.session_state:
                    try:
                        del st.session_state[key]
                    except KeyError: # Handle cases where key might already be deleted
                        pass
            st.rerun()
