import streamlit as st

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

    with st.form("login_form"): # Use a form for better structure
        username = st.text_input("用户名")
        password = st.text_input("密码", type="password")
        submitted = st.form_submit_button("登录")

        if submitted:
            # Placeholder logic: Any input is considered valid for now
            if username and password:
                st.session_state.logged_in = True
                st.rerun() # Rerun the script to hide the form and show the app
            else:
                st.error("请输入用户名和密码")

    return False # Return False if login form is shown or submission failed

def add_logout_button():
    """Adds a logout button to the sidebar if logged in."""
    if st.session_state.get('logged_in', False):
        if st.sidebar.button("退出登录"):
            st.session_state.logged_in = False
            # Clear other relevant session state if needed upon logout
            keys_to_clear = [
                'processing_results', 'formatted_data', 'edited_data',
                'selected_template_option', 'manual_template',
                'loaded_template', 'loaded_template_name', 'edited_loaded_template',
                'input_images' # Add input_images to clear
            ]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
