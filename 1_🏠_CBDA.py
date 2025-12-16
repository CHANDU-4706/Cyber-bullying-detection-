import streamlit as st
import pandas as pd
from pathlib import Path

# Path to the Excel file
user_data_path = "user.xlsx"

# Function to load user data from Excel
def load_user_data():
    if Path(user_data_path).is_file():
        return pd.read_excel(user_data_path)
    else:
        return pd.DataFrame(columns=["Username", "Password"])

# Function to save user data to Excel
def save_user_data(dataframe):
    dataframe.to_excel(user_data_path, index=False)

# Function to check if a user exists
def user_exists(username):
    user_data = load_user_data()
    return not user_data[user_data["Username"] == username].empty

# Function to verify user credentials
def verify_user(username, password):
    user_data = load_user_data()
    user = user_data[(user_data["Username"] == username) & (user_data["Password"] == password)]
    return not user.empty

# Function to add a new user
def add_user(username, password):
    user_data = load_user_data()
    new_user = pd.DataFrame({"Username": [username], "Password": [password]})
    user_data = pd.concat([user_data, new_user], ignore_index=True)
    save_user_data(user_data)
    return True  # Return True if user added successfully

# Function to logout
def logout():
    st.session_state['logged_in'] = False
    st.session_state['username'] = None
    st.experimental_rerun()  # Refresh the page to reflect logout

# Streamlit interface for login/signup
st.title("CBDA Login/Signup")

menu = st.sidebar.selectbox("Menu", ["Login", "Signup", "Logout"])

if menu == "Signup":
    st.subheader("Create a new account")

    new_username = st.text_input("Username")
    new_password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")

    if st.button("Signup"):
        if new_username and new_password and confirm_password:
            if new_password == confirm_password:
                if not user_exists(new_username):
                    add_user(new_username, new_password)
                    st.success("Account created successfully!")
                else:
                    st.error("Username already exists.")
            else:
                st.error("Passwords do not match.")
        else:
            st.error("Please fill out all fields.")

elif menu == "Login":
    st.subheader("Login to your account")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if verify_user(username, password):
            st.session_state['logged_in'] = True
            st.session_state['username'] = username
            st.success("Login successful!")
            st.experimental_rerun()  # Refresh the page to proceed to the main app
        else:
            st.error("Invalid username or password.")

elif menu == "Logout":
    logout()
    st.success("Logged out successfully!")

# Check login status
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if st.session_state['logged_in']:
    st.sidebar.success(f"Logged in as {st.session_state['username']}")
    st.stop()  # Prevent the rest of the code from running