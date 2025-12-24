# ==========================================
# 🐞 FIX 1: SQLITE PATCH (MUST BE TOP)
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
from google.oauth2.service_account import Credentials
from datetime import datetime
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from huggingface_hub import InferenceClient

# ==========================================
# 🔧 CONFIG
# ==========================================
st.set_page_config(page_title="BookBot Pro", page_icon="📚", layout="wide")

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
# ☁️ GOOGLE SHEETS CONNECTION
# ==========================================
@st.cache_resource
def get_client():
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
    if "gcp_service_account" in st.secrets:
        creds = Credentials.from_service_account_info(
            st.secrets["gcp_service_account"], scopes=scopes
        )
    elif os.path.exists("secrets.json"):
        creds = Credentials.from_service_account_file("secrets.json", scopes=scopes)
    else:
        st.error("⚠️ Google Credentials missing!")
        st.stop()
    return gspread.authorize(creds)

def get_sheet(sheet_name):
    try:
        client = get_client()
        return client.open("BookBot_Data").worksheet(sheet_name)
    except Exception as e:
        # Auto-create Reviews tab if missing (Safety feature)
        if sheet_name == "Reviews" and "WorksheetNotFound" in str(type(e)):
            client = get_client()
            sh = client.open("BookBot_Data")
            ws = sh.add_worksheet(title="Reviews", rows="1000", cols="5")
            ws.append_row(["Book Title", "Username", "Rating", "Comment", "Date"])
            return ws
        st.error(f"⚠️ Cloud Error: Could not find tab '{sheet_name}'.")
        st.stop()

# ==========================================
# 🔐 AUTH & DATABASE FUNCTIONS
# ==========================================
def login_user(username, password):
    try:
        users_sheet = get_sheet("Users")
        records = users_sheet.get_all_values()
        for row in records[1:]:
            if len(row) >= 2 and row[0].strip().lower() == username.strip().lower():
                if row[1].strip() == password.strip():
                    return "success"
                else:
                    return "wrong_pass"
        users_sheet.append_row([username, password])
        return "register"
    except: return "error"

def get_wishlist(username):
    try:
        sheet = get_sheet("Sheet1")
        data = sheet.get_all_values()
        return [row[1] for row in data if len(row) > 1 and row[0] == username]
    except: return []

def save_to_wishlist(username, book_title):
    try:
        sheet = get_sheet("Sheet1")
        data = sheet.get_all_values()
        for row in data:
            if len(row) > 1 and row[0] == username and row[1] == book_title:
                st.toast("⚠️ Already in wishlist!", icon="ℹ️")
                return
        sheet.append_row([username, book_title, datetime.now().strftime("%Y-%m-%d")])
        st.toast("✅ Added to Wishlist!", icon="🎉")
    except Exception as e: st.error(f"Save Failed: {e}")

def remove_from_wishlist(username, book_title):
    try:
        sheet = get_sheet("Sheet1")
        data = sheet.get_all_values()
        for i, row in enumerate(data):
            if len(row) > 1 and row[0] == username and row[1] == book_title:
                sheet.delete_rows(i + 1)
                st.toast("🗑️ Removed.", icon="✅")
                return
    except: pass

def save_review(book_title, username, rating, comment):
    try:
        sheet = get_sheet("Reviews")
        sheet.append_row([book_title, username, rating, comment, datetime.now().strftime("%Y-%m-%d")])
        return True
    except Exception as e:
        st.error(f"Review Error: {e}")
        return False

def get_app_reviews(book_title):
    try:
        sheet = get_sheet("Reviews")
        data = sheet.get_all_values()
        # Filter for this book
        return [
            {"User": row[1], "Rating": row[2], "Comment": row[3]} 
            for row in data if len(row) > 3 and row[0] == book_title
        ]
    except: return []

# ==========================================
# 🧠 AI & SEARCH LOGIC
# ==========================================
def search_google(query):
    # ROBUST SEARCH: Forces English, handles missing data
    url = "https://www.googleapis.com/books/v1/volumes"
    params = {"q": query, "maxResults": 1, "langRestrict": "en", "printType": "books"}
    try:
        res = requests.get(url, params=params).json()
        if "items" in res:
            b = res["items"][0]["volumeInfo"]
            title = b.get("title", "Unknown")
            authors = ", ".join(b.get("authors", ["Unknown"]))
            desc = b.get("description", "No description available.")
            rating = b.get("averageRating", 0)
            
            # Safe Image Get
            img = b.get("imageLinks", {}).get("thumbnail")
            
            return {
                "text": f"Title: {title}\nAuthor: {authors}\nDescription: {desc}",
                "title": title,
                "image": img,
                "stars": "⭐"*int(rating) + ("½" if rating % 1 >= 0.5 else "") if rating else "",
                "count": b.get("ratingsCount", 0),
                "link": b.get("previewLink", b.get("infoLink")),
                "found": True
            }
    except: pass
    return {"found": False, "image": None}

def ask_ai_raw(sys_msg, user_msg):
    token = os.environ['HUGGINGFACEHUB_API_TOKEN']
    # Zephyr 7B Beta - The reliable free model
    client = InferenceClient(token=token)
    
    # SYSTEM PROMPT: Enforce brevity and NO SPOILERS
    system_instruction = (
        f"{sys_msg}\n\n"
        "IMPORTANT RULES:\n"
        "1. Keep the answer under 150 words.\n"
        "2. Do NOT reveal the ending or major spoilers.\n"
        "3. Be engaging and helpful."
    )
    
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_msg}
    ]

    try:
        response = client.chat_completion(
            messages=messages,
            model="HuggingFaceH4/zephyr-7b-beta", 
            max_tokens=400,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        if "401" in str(e): return "⚠️ Auth Error: Check Token."
        if "503" in str(e): return "⏳ AI is loading... try again in 30s."
        return f"⚠️ Error: {str(e)}"

# ==========================================
# 📄 PDF LOGIC
# ==========================================
@st.cache_resource
def setup_chroma():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(persist_directory="./library_db", embedding_function=embeddings), embeddings

try:
    vector_db, embeddings = setup_chroma()
except: st.stop()

def process_pdf(f):
    reader = pypdf.PdfReader(f)
    text = "".join([p.extract_text() or "" for p in reader.pages])
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).create_documents([text])
    return Chroma.from_documents(chunks, embeddings)

# ==========================================
# 🖥️ MAIN UI
# ==========================================
def main_app():
    user = st.session_state["username"]
    
    # SIDEBAR
    st.sidebar.title(f"👤 {user}")
    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.rerun()
    st.sidebar.divider()
    st.sidebar.header("❤️ My Wishlist")
    my_list = get_wishlist(user)
    for b in my_list: st.sidebar.write(f"📖 {b}")

    st.title(f"📚 BookBot | Hello, {user}")
    
    t1, t2, t3, t4 = st.tabs(["💬 Chat & Search", "💡 Recommendations", "📖 My Wishlist", "📄 PDF Study"])
    
    # --- TAB 1: SEARCH & CHAT ---
    with t1:
        if "msgs" not in st.session_state: st.session_state.msgs = []
        for m in st.session_state.msgs:
            with st.chat_message(m["role"]):
                st.write(m["content"])
                if m.get("image"): st.image(m["image"], width=130)
        
        if q := st.chat_input("Search for a book..."):
            st.session_state.msgs.append({"role": "user", "content": q})
            st.chat_message("user").write(q)
            
            with st.spinner("Librarian is searching..."):
                scout = search_google(q)
                
                # Construct Prompt based on finding
                if scout["found"]:
                    prompt = f"Give me a brief summary of '{scout['title']}' based on this: {scout['text']}"
                else:
                    prompt = q
                
                ans = ask_ai_raw("You are a helpful Librarian.", prompt)
                
                # Show Result
                st.chat_message("assistant").write(ans)
                
                if scout["found"]:
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        if scout["image"]: st.image(scout["image"], width=130)
                    with col2:
                        st.subheader(scout["title"])
                        if scout["stars"]: st.caption(f"{scout['stars']} ({scout['count']} Google reviews)")
                        
                        # --- THE BUTTONS YOU ASKED FOR ---
                        c_a, c_b = st.columns(2)
                        with c_a:
                            # 1. Wishlist Button
                            st.button("❤️ Add to Wishlist", key=f"save_{scout['title']}_{datetime.now()}", 
                                      on_click=save_to_wishlist, args=(user, scout["title"]))
                        with c_b:
                            # 2. Google Reviews Link
                            if scout["link"]: st.link_button("🔗 Read Google Reviews", scout["link"])
                    
                    # --- 3. WRITE REVIEW SECTION ---
                    st.divider()
                    with st.expander(f"✍️ Write a Review for {scout['title']}"):
                        with st.form(key=f"rev_{scout['title']}"):
                            u_rating = st.slider("Rating", 1, 5, 5)
                            u_comment = st.text_area("Your thoughts...")
                            if st.form_submit_button("Post Review"):
                                if save_review(scout["title"], user, u_rating, u_comment):
                                    st.success("Review Posted!")
                    
                    # Show previous app reviews
                    app_reviews = get_app_reviews(scout["title"])
                    if app_reviews:
                        st.caption("User Reviews:")
                        for r in app_reviews:
                            st.markdown(f"**{r['User']}**: {r['Comment']} ({'⭐'*int(r['Rating'])})")

                # Save history
                st.session_state.msgs.append({"role": "assistant", "content": ans, "image": scout.get("image")})

    # --- TAB 2: RECS ---
    with t2:
        taste = st.text_area("I like...")
        if st.button("Get Ideas"): 
            st.markdown(ask_ai_raw("Recommend 3 books. Keep it brief.", taste))

    # --- TAB 3: WISHLIST ---
    with t3:
        st.header(f"📖 {user}'s Collection")
        if my_list:
            df = pd.DataFrame(my_list, columns=["Title"])
            st.dataframe(df, use_container_width=True)
            to_del = st.selectbox("Select book to remove:", my_list)
            st.button("🗑️ Remove Book", on_click=remove_from_wishlist, args=(user, to_del))
        else: st.info("Your wishlist is empty.")

    # --- TAB 4: PDF ---
    with t4:
        st.header("📄 PDF Analyzer")
        f = st.file_uploader("Upload PDF", type="pdf")
        if f:
            if "pdf_id" not in st.session_state or st.session_state.pdf_id != f.name:
                st.session_state.db = process_pdf(f)
                st.session_state.pdf_id = f.name
                st.success("Analyzed!")
            q_pdf = st.text_input("Ask about the PDF:")
            if st.button("Ask"):
                docs = st.session_state.db.similarity_search(q_pdf, k=3)
                ctx = "\n".join([d.page_content for d in docs])
                st.write(ask_ai_raw(f"Context: {ctx}", q_pdf))

# ==========================================
# 🔐 LOGIN SCREEN
# ==========================================
if "username" not in st.session_state:
    st.title("🔐 BookBot Secure Login")
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        with st.form("auth"):
            st.info("New users registered automatically.")
            u = st.text_input("Username").strip()
            p = st.text_input("Password", type="password").strip()
            if st.form_submit_button("Enter"):
                if u and p:
                    res = login_user(u, p)
                    if res == "success":
                        st.session_state["username"] = u
                        st.rerun()
                    elif res == "register":
                        st.session_state["username"] = u
                        st.success("Account Created!")
                        st.rerun()
                    else: st.error("Wrong Password.")
                else: st.warning("Fill all fields.")
else:
    main_app()
