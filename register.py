import streamlit as st
import os

st.set_page_config(page_title="Register | Smart SNA Dashboard")

st.title("📝 Register")

with st.form("register_form"):
    username = st.text_input("Choose a Username")
    password = st.text_input("Choose a Password", type="password")
    submitted = st.form_submit_button("Register")

    if submitted:
        if not username or not password:
            st.warning("⚠️ Username and password cannot be empty")
        else:
            # Check if user exists
            users = {}
            if os.path.exists("users.csv"):
                with open("users.csv", "r") as f:
                    for line in f.readlines():
                        u, p = line.strip().split(",")
                        users[u] = p

            if username in users:
                st.error("❌ Username already exists")
            else:
                # Save new user
                with open("users.csv", "a") as f:
                    f.write(f"{username},{password}\n")
                st.success(f"✅ User {username} registered successfully!")
                st.info("Go to the Login page to access the dashboard")
