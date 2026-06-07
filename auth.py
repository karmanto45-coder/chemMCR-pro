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
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');
    html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}

    /* Hide GitHub link, footer, and deploy button */
    #MainMenu {visibility:hidden;}
    footer {visibility:hidden;}
    [data-testid="stToolbar"] {visibility:hidden;}
    a[href*="github"] {display:none !important;}
    .stDeployButton {display:none !important;}

    /* Full page dark background */
    .stApp { background: #080d14; }
    [data-testid="stSidebar"] { display: none; }

    /* Login card */
    .login-outer {
        display: flex; flex-direction: column;
        align-items: center; justify-content: center;
        min-height: 88vh; padding: 2rem;
    }
    .login-card {
        background: linear-gradient(145deg, #0f1520, #141e2e);
        border: 1px solid #1e3a5f;
        border-radius: 20px;
        padding: 2.8rem 2.5rem 2rem;
        width: 100%; max-width: 420px;
        box-shadow: 0 0 60px rgba(56,139,253,0.08), 0 20px 40px rgba(0,0,0,0.4);
    }

    /* Logo area */
    .logo-icon {
        font-size: 2.8rem; text-align: center;
        margin-bottom: 0.5rem; line-height: 1;
    }
    .login-logo {
        font-family: 'DM Mono', monospace;
        font-size: 2rem; font-weight: 500;
        text-align: center; margin: 0;
        letter-spacing: -1px;
        background: linear-gradient(135deg, #7dd3fc, #3b82f6, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .login-sub {
        font-size: 0.78rem; color: #475569;
        text-align: center; margin: 6px 0 1.5rem;
        letter-spacing: 0.03em;
    }
    .login-version {
        display: block; text-align: center;
        font-size: 0.68rem; background: #0d1829;
        border: 1px solid #1e3a5f;
        color: #7dd3fc; padding: 3px 12px;
        border-radius: 20px; margin: 0 auto 1.8rem;
        font-family: 'DM Mono', monospace;
        width: fit-content;
    }
    .login-divider {
        border: none; border-top: 1px solid #1e293b;
        margin: 1.5rem 0 0.5rem;
    }
    .login-footer {
        text-align: center; font-size: 0.72rem;
        color: #334155; margin-top: 1.2rem;
    }
    </style>

    <div class="login-outer">
      <div class="login-card">
        <div class="logo-icon">🔬</div>
        <p class="login-logo">SpectraID Pro</p>
        <p class="login-sub">MULTIVARIATE CURVE RESOLUTION &nbsp;·&nbsp; SPECTRAL IDENTIFICATION</p>
        <span class="login-version">v2.0 &nbsp;·&nbsp; Professional Edition</span>
      </div>
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

            st.markdown("""
            <div style='text-align:center;margin-top:1rem;font-size:0.75rem;color:#475569;'>
            Default: admin / admin123<br>
            Ganti password setelah login pertama
            </div>
            """, unsafe_allow_html=True)
