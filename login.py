import streamlit as st

st.set_page_config(page_title="Login | Smart SNA Dashboard")

# Initialize session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

# --- Login Form ---
st.title("🔐 Login")
with st.form("login_form"):
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    submitted = st.form_submit_button("Login")

    if submitted:
        # Simple login check (replace with DB or secure check in production)
        try:
            users = {}
            with open("users.csv", "r") as f:
                for line in f.readlines():
                    u, p = line.strip().split(",")
                    users[u] = p
        except FileNotFoundError:
            users = {}

        if username in users and users[username] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success(f"✅ Logged in as {username}")
            st.experimental_rerun()  # reload app.py
        else:
            st.error("❌ Invalid username or password")

st.info("Don't have an account? Go to the Register page.")
