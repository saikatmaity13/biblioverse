# ==========================================
# 🐞 FIX 1: SQLITE PATCH
# ==========================================
import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# ==========================================
# 📦 IMPORTS
# ==========================================
import os
import ssl
import pandas as pd
import requests
import streamlit as st
import pypdf
import gspread
from google.oauth2.service_account import Credentials # <--- NEW MODERN LIBRARY
from datetime import datetime
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ==========================================
# 🔧 CONFIG
# ==========================================
st.set_page_config(page_title="Biblioverse",  layout="wide")

# Fix SSL
os.environ['HF_HUB_DISABLE_SSL_VERIFY'] = '1'
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Security: Load Token
if "HUGGINGFACE_TOKEN" in st.secrets:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACE_TOKEN"]
else:
    st.error("⚠️ HuggingFace Token missing in Secrets!")
    st.stop()

# ==========================================
# ☁️ GOOGLE SHEETS (NEW MODERN CONNECTION)
# ==========================================
@st.cache_resource
def get_client():
    # Define the Scopes (Permissions)
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
    
    # 1. Try Loading from Streamlit Secrets (Cloud)
    if "gcp_service_account" in st.secrets:
        creds = Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=scopes
        )
    # 2. Try Loading from Local File (Local Testing)
    elif os.path.exists("secrets.json"):
        creds = Credentials.from_service_account_file("secrets.json", scopes=scopes)
    else:
        st.error("⚠️ Google Credentials missing!")
        st.stop()
        
    return gspread.authorize(creds)

def get_sheet(sheet_name):
    """Helper to get a specific tab."""
    try:
        client = get_client()
        # Open by Name
        return client.open("BookBot_Data").worksheet(sheet_name)
    except Exception as e:
        # If "Users" tab is missing, we catch it here
        if "WorksheetNotFound" in str(type(e)):
            st.error(f"⚠️ Critical Error: The tab '{sheet_name}' was not found in your Google Sheet.")
            st.info("Please create a new tab named exactly 'Users' in your sheet.")
            st.stop()
        elif "SpreadsheetNotFound" in str(type(e)):
            st.error("⚠️ Critical Error: The Google Sheet 'BookBot_Data' was not found.")
            st.info(f"Make sure you shared the sheet with this email: {st.secrets['gcp_service_account']['client_email']}")
            st.stop()
        raise e

# ==========================================
# 🔐 AUTHENTICATION SYSTEM
# ==========================================
def login_user(username, password):
    """
    Returns: "success", "wrong_pass", "register", "error"
    """
    try:
        users_sheet = get_sheet("Users")
        records = users_sheet.get_all_values()
        
        # Check if user exists (Skip header row)
        for row in records[1:]:
            if len(row) >= 2 and row[0].strip().lower() == username.strip().lower():
                # User Found: Check Password
                if row[1].strip() == password.strip():
                    return "success"
                else:
                    return "wrong_pass"
        
        # User Not Found: Register them
        users_sheet.append_row([username, password])
        return "register"
    except Exception as e:
        st.error(f"Auth Error: {e}")
        return "error"

# ==========================================
# ☁️ WISHLIST FUNCTIONS
# ==========================================
def get_my_books(username):
    try:
        sheet = get_sheet("Sheet1") # Default tab for books
        data = sheet.get_all_values()
        return [row[1] for row in data if len(row) > 1 and row[0] == username]
    except: return []

def save_book_to_cloud(username, book_title):
    try:
        sheet = get_sheet("Sheet1")
        data = sheet.get_all_values()
        for row in data:
            if len(row) > 1 and row[0] == username and row[1] == book_title:
                st.toast("⚠️ Already saved!", icon="ℹ️")
                return

        date_str = datetime.now().strftime("%Y-%m-%d")
        sheet.append_row([username, book_title, date_str])
        st.toast(f"✅ Saved!", icon="🎉")
    except Exception as e:
        st.error(f"Save Error: {e}")

def remove_book_from_cloud(username, book_title):
    try:
        sheet = get_sheet("Sheet1")
        data = sheet.get_all_values()
        for i, row in enumerate(data):
            if len(row) > 1 and row[0] == username and row[1] == book_title:
                sheet.delete_rows(i + 1)
                st.toast("🗑️ Removed.", icon="✅")
                return
    except: pass

# ==========================================
# 🧠 AI & UTILS
# ==========================================
@st.cache_resource
def setup_brain():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(persist_directory="./library_db", embedding_function=embeddings)
    return db, embeddings

try:
    vector_db, embeddings = setup_brain()
except: st.stop()

def ask_ai_raw(sys_msg, user_msg):
    """
    Uses the official Hugging Face Client to connect to the best available free model.
    """
    token = os.environ['HUGGINGFACEHUB_API_TOKEN']
    
    # We use Zephyr because it is the most reliable UNGATED free model.
    # (Mistral and Gemma often require you to accept terms on the website, causing errors)
    repo_id = "HuggingFaceH4/zephyr-7b-beta"
    
    client = InferenceClient(token=token)
    
    # Create the prompt in the format Zephyr expects
    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user_msg}
    ]

    try:
        # The client automatically handles the new 'router' URLs for you
        response = client.chat_completion(
            messages=messages,
            model=repo_id, 
            max_tokens=500,
            temperature=0.7
        )
        
        # Return the clean text
        return response.choices[0].message.content

    except Exception as e:
        error_msg = str(e)
        
        # specific help for common errors
        if "401" in error_msg:
            return "⚠️ Auth Error: Your Hugging Face Token is invalid. Please generate a new one."
        if "429" in error_msg:
            return "⏳ Busy: You reached the hourly limit. Please wait a few minutes."
        if "503" in error_msg:
            return "⏳ The Model is loading. Please wait 30 seconds and try again."
            
        return f"⚠️ Error: {error_msg}"
        
    

def process_pdf(f):
    reader = pypdf.PdfReader(f)
    text = "".join([p.extract_text() or "" for p in reader.pages])
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).create_documents([text])
    return Chroma.from_documents(chunks, embeddings)

def search_google(query):
    try:
        res = requests.get("https://www.googleapis.com/books/v1/volumes", params={"q": query, "maxResults": 1}).json()
        if "items" in res:
            b = res["items"][0]["volumeInfo"]
            rate = b.get("averageRating", 0)
            return {
                "text": f"Title: {b.get('title')}\nAuthor: {b.get('authors')}\nDesc: {b.get('description')}",
                "title": b.get("title"),
                "image": b.get("imageLinks", {}).get("thumbnail"),
                "stars": "⭐"*int(rate) if rate else "",
                "count": b.get("ratingsCount", 0),
                "link": b.get("previewLink"),
                "found": True
            }
    except: pass
    return {"found": False, "image": None}

# ==========================================
# 🖥️ MAIN APP LOGIC
# ==========================================
def main_app():
    user = st.session_state["username"]
    st.sidebar.title(f"👤 {user}")
    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.rerun()
    
    st.sidebar.divider()
    st.sidebar.header("❤️ Wishlist")
    my_books = get_my_books(user)
    for b in my_books: st.sidebar.write(f"📖 {b}")

    st.title(f"📚 BookBot | Hi, {user}")
    
    t1, t2, t3, t4 = st.tabs(["💬 Chat", "💡 Recs", "📖 Wishlist", "📄 PDF Study"])
    
    with t1:
        if "msgs" not in st.session_state: st.session_state.msgs = []
        for m in st.session_state.msgs:
            with st.chat_message(m["role"]):
                st.write(m["content"])
                if m.get("image"): st.image(m["image"], width=130)
        
        if q := st.chat_input("Ask about a book..."):
            st.session_state.msgs.append({"role": "user", "content": q})
            st.chat_message("user").write(q)
            
            with st.spinner("Searching..."):
                scout = search_google(q)
                sys = f"Answer based on: {scout['text']}" if scout["found"] else "Answer generally."
                ans = ask_ai_raw(sys, q)
                
                st.chat_message("assistant").write(ans)
                if scout["found"]:
                    c1, c2 = st.columns([1,3])
                    with c1: 
                        if scout["image"]: st.image(scout["image"], width=130)
                    with c2:
                        if scout["stars"]: st.caption(f"{scout['stars']} ({scout['count']} reviews)")
                        st.button("❤️ Save to My Wishlist", key=f"save_{scout['title']}", 
                                  on_click=save_book_to_cloud, args=(user, scout["title"]))
                        if scout["link"]: st.link_button("🔗 Google Reviews", scout["link"])
                
                st.session_state.msgs.append({"role": "assistant", "content": ans, "image": scout.get("image")})

    with t2:
        taste = st.text_area("I like...")
        if st.button("Get Ideas"): st.markdown(ask_ai_raw("Recommend 3 books.", taste))

    with t3:
        st.header(f"📖 {user}'s Collection")
        if my_books:
            df = pd.DataFrame(my_books, columns=["Title"])
            st.dataframe(df, use_container_width=True)
            to_del = st.selectbox("Remove:", my_books)
            st.button("🗑️ Remove", on_click=remove_book_from_cloud, args=(user, to_del))
        else: st.info("Empty.")

    with t4:
        st.header("📄 PDF Analyzer")
        f = st.file_uploader("Upload PDF", type="pdf")
        if f:
            if "pdf_id" not in st.session_state or st.session_state.pdf_id != f.name:
                st.session_state.db = process_pdf(f)
                st.session_state.pdf_id = f.name
                st.success("Ready!")
            q_pdf = st.text_input("Ask PDF:")
            if st.button("Analyze"):
                docs = st.session_state.db.similarity_search(q_pdf, k=3)
                ctx = "\n".join([d.page_content for d in docs])
                st.write(ask_ai_raw(f"Context: {ctx}", q_pdf))

# ==========================================
# 🔐 AUTH SCREEN
# ==========================================
if "username" not in st.session_state:
    st.title("🔐 BookBot Secure Login")
    
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        with st.form("auth_form"):
            st.info("New users will be registered automatically.")
            u = st.text_input("Username").strip()
            p = st.text_input("Password", type="password").strip()
            submitted = st.form_submit_button("Login / Register")
            
            if submitted:
                if u and p:
                    with st.spinner("Verifying credentials..."):
                        status = login_user(u, p)
                        if status == "success":
                            st.session_state["username"] = u
                            st.success(f"Welcome back, {u}!")
                            st.rerun()
                        elif status == "register":
                            st.session_state["username"] = u
                            st.success(f"Account created for {u}!")
                            st.rerun()
                        elif status == "wrong_pass":
                            st.error("❌ Incorrect Password! That username is already taken.")
                        else:
                            st.error("System Error. Try again.")
                else:
                    st.warning("Please enter username and password.")
else:
    main_app()








