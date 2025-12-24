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
# 🔧 CONFIG & SECURITY
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

# Load Token
if "HUGGINGFACE_TOKEN" in st.secrets:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACE_TOKEN"]
else:
    st.error("⚠️ HuggingFace Token missing in Secrets!")
    st.stop()

# ==========================================
# ☁️ DATABASE (GOOGLE SHEETS)
# ==========================================
@st.cache_resource
def get_client():
    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    if "gcp_service_account" in st.secrets:
        creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scopes)
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
        # Auto-create Reviews tab if missing
        if sheet_name == "Reviews":
            client = get_client()
            sh = client.open("BookBot_Data")
            ws = sh.add_worksheet(title="Reviews", rows="1000", cols="5")
            ws.append_row(["Book Title", "Username", "Rating", "Comment", "Date"])
            return ws
        st.error(f"⚠️ Cloud Error: Tab '{sheet_name}' not found.")
        st.stop()

# --- DB Functions ---
def login_user(username, password):
    try:
        users_sheet = get_sheet("Users")
        records = users_sheet.get_all_values()
        for row in records[1:]:
            if len(row) >= 2 and row[0].strip().lower() == username.strip().lower():
                return "success" if row[1].strip() == password.strip() else "wrong_pass"
        users_sheet.append_row([username, password])
        return "register"
    except: return "error"

def get_wishlist(username):
    try:
        sheet = get_sheet("Sheet1")
        return [row[1] for row in sheet.get_all_values() if len(row) > 1 and row[0] == username]
    except: return []

def save_to_wishlist(username, book_title):
    try:
        sheet = get_sheet("Sheet1")
        for row in sheet.get_all_values():
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
        return [{"User": r[1], "Rating": r[2], "Comment": r[3]} for r in sheet.get_all_values() if len(r) > 3 and r[0] == book_title]
    except: return []

# ==========================================
# 🧠 AI & SEARCH
# ==========================================
def search_google(query):
    # Searches for a book and gets details
    url = "https://www.googleapis.com/books/v1/volumes"
    params = {"q": query, "maxResults": 1, "langRestrict": "en", "printType": "books"}
    try:
        res = requests.get(url, params=params).json()
        if "items" in res:
            b = res["items"][0]["volumeInfo"]
            title = b.get("title", "Unknown")
            return {
                "text": f"Title: {title}\nAuthor: {', '.join(b.get('authors', []))}\nDesc: {b.get('description', '')}",
                "title": title,
                "image": b.get("imageLinks", {}).get("thumbnail"),
                "stars": "⭐"*int(b.get("averageRating", 0)),
                "count": b.get("ratingsCount", 0),
                "link": b.get("previewLink", b.get("infoLink")),
                "found": True
            }
    except: pass
    return {"found": False, "image": None}

def ask_ai_summary(book_info):
    token = os.environ['HUGGINGFACEHUB_API_TOKEN']
    client = InferenceClient(token=token)
    
    # SYSTEM PROMPT: Forces a brief summary
    sys_msg = (
        "You are a helpful librarian. "
        "Provide a BRIEF summary (max 3-4 sentences) of this book. "
        "Do NOT spoil the ending. Focus on the plot setup and themes."
    )
    
    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": f"Book Details:\n{book_info}"}
    ]

    try:
        response = client.chat_completion(
            messages=messages,
            model="HuggingFaceH4/zephyr-7b-beta", 
            max_tokens=300,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return "⚠️ I couldn't generate a summary right now."

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

def ask_pdf(question):
    # Simple Q&A for PDF
    docs = vector_db.similarity_search(question, k=3)
    ctx = "\n".join([d.page_content for d in docs])
    client = InferenceClient(token=os.environ['HUGGINGFACEHUB_API_TOKEN'])
    msgs = [{"role": "system", "content": f"Answer based on:\n{ctx}"}, {"role": "user", "content": question}]
    try:
        return client.chat_completion(messages=msgs, model="HuggingFaceH4/zephyr-7b-beta", max_tokens=400).choices[0].message.content
    except: return "Error analyzing PDF."

# ==========================================
# 🖥️ MAIN UI
# ==========================================
def main_app():
    user = st.session_state["username"]
    
    # Sidebar
    st.sidebar.title(f"👤 {user}")
    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.rerun()
    st.sidebar.divider()
    st.sidebar.header("❤️ My Wishlist")
    for b in get_wishlist(user): st.sidebar.write(f"📖 {b}")

    st.title(f"📚 BookBot | Welcome, {user}")
    
    t1, t2, t3, t4 = st.tabs(["💬 Chat & Search", "💡 Recommendations", "📖 Wishlist", "📄 Study Room"])
    
    # --- TAB 1: SEARCH & CHAT ---
    with t1:
        st.info("💡 Type a book name below to get a summary!")
        
        # Chat History container
        chat_container = st.container()
        
        # Input at bottom
        if query := st.chat_input("Enter book name..."):
            
            # 1. Search Google Books
            scout = search_google(query)
            
            # 2. Generate AI Summary
            if scout["found"]:
                summary = ask_ai_summary(scout["text"])
            else:
                summary = "I couldn't find that book. Try checking the spelling?"

            # 3. Store in session state (History)
            if "history" not in st.session_state: st.session_state.history = []
            st.session_state.history.append({
                "role": "user", "content": query
            })
            st.session_state.history.append({
                "role": "assistant", 
                "content": summary, 
                "book_data": scout if scout["found"] else None
            })

        # Display History
        with chat_container:
            if "history" in st.session_state:
                for msg in st.session_state.history:
                    with st.chat_message(msg["role"]):
                        st.write(msg["content"])
                        
                        # IF IT'S A BOOK RESULT, SHOW THE INTERACTIVE CARD
                        if msg.get("book_data"):
                            b = msg["book_data"]
                            
                            # Layout: Image Left, Details Right
                            c1, c2 = st.columns([1, 3])
                            with c1:
                                if b["image"]: st.image(b["image"], width=120)
                            with c2:
                                st.subheader(b["title"])
                                if b["stars"]: st.caption(f"Google Rating: {b['stars']} ({b['count']} votes)")
                                
                                # BUTTONS ROW
                                b1, b2 = st.columns(2)
                                with b1:
                                    # ACTION 1: ADD TO WISHLIST
                                    st.button("❤️ Add to Wishlist", key=f"add_{b['title']}_{len(st.session_state.history)}", 
                                              on_click=save_to_wishlist, args=(user, b["title"]))
                                with b2:
                                    if b["link"]: st.link_button("🔗 Google Reviews", b["link"])

                            # ACTION 2: WRITE REVIEW (Expandable Form)
                            with st.expander(f"✍️ Write a Review for {b['title']}"):
                                with st.form(key=f"review_{b['title']}_{len(st.session_state.history)}"):
                                    st.write("What did you think?")
                                    u_rating = st.slider("Rating", 1, 5, 5)
                                    u_comment = st.text_area("Your review...")
                                    if st.form_submit_button("Submit Review"):
                                        if save_review(b["title"], user, u_rating, u_comment):
                                            st.success("Review Saved!")

    # --- TAB 2: RECS ---
    with t2:
        t = st.text_area("I enjoy books like...")
        if st.button("Get Ideas"): 
            client = InferenceClient(token=os.environ['HUGGINGFACEHUB_API_TOKEN'])
            res = client.chat_completion(messages=[{"role":"user", "content": f"Recommend 3 books based on: {t}"}], model="HuggingFaceH4/zephyr-7b-beta", max_tokens=300)
            st.write(res.choices[0].message.content)

    # --- TAB 3: WISHLIST ---
    with t3:
        st.header(f"📖 {user}'s Collection")
        my_list = get_wishlist(user)
        if my_list:
            st.dataframe(pd.DataFrame(my_list, columns=["Title"]), use_container_width=True)
            to_del = st.selectbox("Select to remove:", my_list)
            st.button("🗑️ Remove", on_click=remove_from_wishlist, args=(user, to_del))
        else: st.info("Empty.")

    # --- TAB 4: PDF ---
    with t4:
        st.header("📄 PDF Analyzer")
        f = st.file_uploader("Upload PDF", type="pdf")
        if f:
            if "pdf_name" not in st.session_state or st.session_state.pdf_name != f.name:
                st.session_state.db = process_pdf(f)
                st.session_state.pdf_name = f.name
                st.success("Loaded!")
            q = st.text_input("Ask PDF:")
            if st.button("Analyze"): st.write(ask_pdf(q))

# ==========================================
# 🔐 LOGIN
# ==========================================
if "username" not in st.session_state:
    st.title("🔐 BookBot Login")
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        with st.form("login"):
            st.info("New users registered automatically.")
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.form_submit_button("Enter"):
                if u and p:
                    res = login_user(u, p)
                    if res == "success":
                        st.session_state["username"] = u
                        st.rerun()
                    elif res == "register":
                        st.session_state["username"] = u
                        st.success("Registered!")
                        st.rerun()
                    else: st.error("Wrong Password.")
else:
    main_app()
