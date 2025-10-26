import streamlit as st
from supabase import create_client, Client

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Authentication functions
def sign_up(email, password):
    try:
        return supabase.auth.sign_up({"email": email, "password": password})
    except Exception as e:
        st.error(f"Registration failed: {e}")


def sign_in(email, password):
    try:
        return supabase.auth.sign_in_with_password({"email": email, "password": password})
    except Exception as e:
        st.error(f"Login failed: {e}")


def sign_out():
    try:
        supabase.auth.sign_out()
        st.session_state.clear()
        st.session_state.screen = "login"
        st.rerun()
    except Exception as e:
        st.error(f"Logout failed: {e}")


def register_screen():
    st.header("Sign Up", anchor=False)
    username = st.text_input("Username")
    email = st.text_input("Email", key="reg_email") # Unique Key to distinguish between registration page and login page
    password = st.text_input("Password", type="password", key="reg_pass") # Unique Key to distinguish between registration page and login page

    if st.button("Register"):
        if not username or not email or not password:
            st.error("Please fill in all fields.")
            return

        user = sign_up(email, password)
        if user and user.user:
            try:
                supabase.table("profiles").insert({
                    "id": user.user.id,
                    "username": username,
                }).execute()

                existing_stats = supabase.table("profiles_stats").select("user_id").eq("user_id",
                                                                                       user.user.id).execute()
                if not existing_stats.data:
                    supabase.table("profiles_stats").insert({
                        "user_id": user.user.id,
                        "total_classifications": 0,
                        "recycle_count": 0,
                        "compost_count": 0,
                        "trash_count": 0,
                        "carbon_saved_kg": 0
                    }).execute()

                st.success("Registration successful! Please log in.")
                st.session_state.screen = "login"
                st.rerun()
            except Exception as e:
                st.error(f"Failed to create profile: {e}")
        else:
            st.error("Registration failed. Please try again.")

    if st.button("Go to Login >"):
        st.session_state.screen = "login"
        st.rerun()


def login_screen():
    st.header("Login", anchor=False)
    email = st.text_input("Email", key="log_email")
    password = st.text_input("Password", type="password", key="log_pass")

    if st.button("Login"):
        user = sign_in(email, password)
        if user and user.user:
            st.session_state.user_id = user.user.id
            profile = (
                supabase.table("profiles")
                .select("username")
                .eq("id", user.user.id)
                .single()
                .execute()
            )

            if profile.data:
                st.session_state.username = profile.data["username"]
            else:
                st.session_state.username = "User"

            try:
                existing_stats = (
                    supabase.table("profiles_stats")
                    .select("user_id")
                    .eq("user_id", user.user.id)
                    .execute()
                )
                if not existing_stats.data:
                    supabase.table("profiles_stats").insert({
                        "user_id": user.user.id,
                        "total_classifications": 0,
                        "recycle_count": 0,
                        "compost_count": 0,
                        "trash_count": 0,
                        "carbon_saved_kg": 0
                    }).execute()
            except Exception as e:
                st.warning(f"Could not verify or create stats record: {e}")

            st.session_state.screen = "main"
            st.rerun()

    if st.button("Continue Without Account"):
        st.session_state.user_id = None  # Not logged in
        st.session_state.username = "Guest"
        st.session_state.screen = "main"
        st.rerun()

    if st.button("Go to Register >"):
        st.session_state.screen = "register"
        st.rerun()

def is_logged_in():
    return st.session_state.get("user_id") is not None
