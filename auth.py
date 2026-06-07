import hashlib
import json
import os
import streamlit as st

USERS_FILE = "users.json"

DEFAULT_USERS = {
    "admin": {
        "password": hashlib.sha256("admin123".encode()).hexdigest(),
        "role": "admin",
        "name": "Administrator"
    }
}

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE) as f:
            return json.load(f)
    save_users(DEFAULT_USERS)
    return DEFAULT_USERS

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)

def hash_password(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def verify_login(username, password):
    users = load_users()
    if username in users:
        if users[username]["password"] == hash_password(password):
            return users[username]
    return None

def get_role():
    return st.session_state.get("role", None)

def is_admin():
    return st.session_state.get("role") == "admin"

def is_logged_in():
    return st.session_state.get("logged_in", False)

def logout():
    for key in ["logged_in", "role", "username", "display_name"]:
        if key in st.session_state:
            del st.session_state[key]

def render_login():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');
    html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
    .login-wrap{max-width:400px;margin:4rem auto 0;}
    .login-logo{font-family:'DM Mono',monospace;font-size:1.8rem;font-weight:500;
        color:#e2e8f0;margin:0;letter-spacing:-1px;}
    .login-sub{font-size:0.85rem;color:#64748b;margin:4px 0 2rem;}
    .login-version{display:inline-block;font-size:0.7rem;background:#1e293b;
        color:#7dd3fc;padding:2px 8px;border-radius:4px;margin-bottom:1.5rem;
        font-family:'DM Mono',monospace;}
    </style>
    <div class="login-wrap">
      <p class="login-logo">SpectraID Pro</p>
      <p class="login-sub">Multivariate Curve Resolution · Spectral Identification</p>
      <span class="login-version">v2.0 · Professional Edition</span>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            lang = st.selectbox("", ["🇮🇩 Bahasa Indonesia", "🇬🇧 English"],
                label_visibility="collapsed")
            is_en = "English" in lang

            username = st.text_input(
                "Username" if is_en else "Nama pengguna",
                placeholder="Enter username" if is_en else "Masukkan username"
            )
            password = st.text_input(
                "Password" if is_en else "Kata sandi",
                type="password",
                placeholder="Enter password" if is_en else "Masukkan kata sandi"
            )

            if st.button("Login", use_container_width=True):
                if not username or not password:
                    st.error("Please fill in all fields." if is_en else "Harap isi semua kolom.")
                else:
                    user = verify_login(username, password)
                    if user:
                        st.session_state["logged_in"] = True
                        st.session_state["username"] = username
                        st.session_state["role"] = user["role"]
                        st.session_state["display_name"] = user["name"]
                        st.session_state["lang"] = "en" if is_en else "id"
                        st.rerun()
                    else:
                        st.error("Invalid username or password." if is_en else
                                 "Username atau kata sandi salah.")

