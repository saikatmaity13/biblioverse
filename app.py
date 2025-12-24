# ==========================================
# 🐞 SQLITE PATCH (MUST BE TOP)
# ==========================================
import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# ==========================================
# 📦 IMPORTS
# ==========================================
import os
import ssl
import hashlib
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
from typing import Dict, List, Optional, Tuple

# ==========================================
# 🔧 CONFIG & SECURITY
# ==========================================
st.set_page_config(page_title="BookBot Pro", page_icon="📚", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .stButton>button {border-radius: 5px; font-weight: 600;}
    .metric-card {background: #f0f2f6; padding: 15px; border-radius: 8px;}
</style>
""", unsafe_allow_html=True)

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
    st.error("⚠️ HuggingFace Token missing!")
    st.stop()

# ==========================================
# 🔐 SECURITY
# ==========================================
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

# ==========================================
# ☁️ DATABASE MANAGER (AUTO-FIXING)
# ==========================================
@st.cache_resource
def get_sheets_client():
    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    if "gcp_service_account" in st.secrets:
        creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scopes)
    elif os.path.exists("secrets.json"):
        creds = Credentials.from_service_account_file("secrets.json", scopes=scopes)
    else:
        st.error("⚠️ Credentials missing!")
        st.stop()
    return gspread.authorize(creds)

def get_sheet(name: str):
    """
    Tries to open a sheet. If missing, CREATES it with correct headers.
    This fixes the 'Wishlist not working' error.
    """
    client = get_sheets_client()
    try:
        return client.open("BookBot_Data").worksheet(name)
    except:
        # Auto-create missing tab
        sh = client.open("BookBot_Data")
        ws = sh.add_worksheet(title=name, rows="1000", cols="5")
        
        # Define headers for each type
        headers = {
            "Reviews": ["Book Title", "Username", "Rating", "Comment", "Date"],
            "Users": ["Username", "Password", "JoinDate"],
            "Wishlist": ["Username", "Book Title", "Date Added"]  # <--- NEW DEDICATED TAB
        }
        
        if name in headers:
            ws.append_row(headers[name])
        return ws

# ==========================================
# 👤 USER MANAGEMENT
# ==========================================
def login_user(username: str, password: str) -> Tuple[str, str]:
    try:
        sheet = get_sheet("Users")
        records = sheet.get_all_values()[1:]
        
        for row in records:
            if len(row) >= 2 and row[0].lower() == username.lower():
                if row[1] == hash_password(password):
                    return "success", "Login successful!"
                return "wrong_pass", "Incorrect password"
        
        sheet.append_row([username, hash_password(password), datetime.now().strftime("%Y-%m-%d")])
        return "register", "Account created!"
    except Exception as e:
        return "error", str(e)

def get_user_stats(username: str) -> Dict:
    try:
        wishlist_data = get_sheet("Wishlist").get_all_values() # Changed to 'Wishlist'
        review_data = get_sheet("Reviews").get_all_values()
        
        w_count = len([r for r in wishlist_data if len(r) > 1 and r[0] == username])
        r_count = len([r for r in review_data if len(r) > 1 and r[1] == username])
        
        return {"wishlist": w_count, "reviews": r_count}
    except:
        return {"wishlist": 0, "reviews": 0}

# ==========================================
# 📚 BOOK SEARCH
# ==========================================
def search_books(query: str) -> List[Dict]:
    url = "https://www.googleapis.com/books/v1/volumes"
    api_key = st.secrets.get("GOOGLE_BOOKS_KEY")
    
    # Force 1 Result + India
    params = {"q": query, "maxResults": 1, "langRestrict": "en", "country": "IN"}
    if api_key: params["key"] = api_key
    
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if "error" in data:
            st.error(f"⚠️ API Error: {data['error']['message']}")
            return []
            
        books = []
        if "items" in data:
            for item in data["items"]:
                info = item.get("volumeInfo", {})
                books.append({
                    "title": info.get("title", "Unknown"),
                    "authors": ", ".join(info.get("authors", ["Unknown"])),
                    "description": info.get("description", "No description"),
                    "image": info.get("imageLinks", {}).get("thumbnail"),
                    "rating": info.get("averageRating", 0),
                    "rating_count": info.get("ratingsCount", 0),
                    "link": info.get("previewLink", info.get("infoLink")),
                    "publisher": info.get("publisher", "Unknown"),
                    "date": info.get("publishedDate", "Unknown"),
                    "pages": info.get("pageCount", 0),
                    "categories": ", ".join(info.get("categories", ["General"]))
                })
        return books
    except Exception as e:
        st.error(f"⚠️ Search Error: {str(e)}")
        return []

# ==========================================
# 🤖 AI ASSISTANT
# ==========================================
class AIHelper:
    def __init__(self):
        self.client = InferenceClient(token=os.environ['HUGGINGFACEHUB_API_TOKEN'])
        self.model = "HuggingFaceH4/zephyr-7b-beta"
    
    def summarize_book(self, book: Dict) -> str:
        prompt = f"Title: {book['title']}\nAuthor: {book['authors']}\nDesc: {book['description'][:400]}\n\nBrief 3-sentence summary (no spoilers):"
        try:
            res = self.client.chat_completion(messages=[{"role": "user", "content": prompt}], model=self.model, max_tokens=200)
            return res.choices[0].message.content
        except: return "Summary unavailable"
    
    def recommend_books(self, preferences: str) -> str:
        try:
            res = self.client.chat_completion(messages=[{"role": "user", "content": f"Recommend 5 books for: {preferences}"}], model=self.model, max_tokens=500)
            return res.choices[0].message.content
        except: return "Recommendations unavailable"
    
    def answer_pdf(self, context: str, question: str) -> str:
        try:
            res = self.client.chat_completion(messages=[{"role": "system", "content": f"Answer based on:\n{context}"}, {"role": "user", "content": question}], model=self.model, max_tokens=400)
            return res.choices[0].message.content
        except: return "Error processing question"

# ==========================================
# 📄 PDF PROCESSOR
# ==========================================
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def process_pdf(file) -> Optional[Chroma]:
    try:
        reader = pypdf.PdfReader(file)
        text = "\n".join([p.extract_text() or "" for p in reader.pages])
        if len(text.strip()) < 100: return None
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.create_documents([text])
        return Chroma.from_documents(chunks, get_embeddings())
    except: return None

# ==========================================
# 📖 WISHLIST (FIXED)
# ==========================================
def get_wishlist(username: str) -> List[Dict]:
    try:
        # Uses dedicated 'Wishlist' tab now
        data = get_sheet("Wishlist").get_all_values()[1:]
        return [{"title": r[1], "date": r[2] if len(r) > 2 else "Unknown"} 
                for r in data if len(r) > 1 and r[0] == username]
    except Exception as e:
        print(f"Wishlist Error: {e}")
        return []

def add_to_wishlist(username: str, title: str):
    """Adds book and forces a rerun to update UI"""
    try:
        sheet = get_sheet("Wishlist")
        existing = sheet.get_all_values()
        
        # Check duplicates
        for row in existing:
            if len(row) > 1 and row[0] == username and row[1] == title:
                st.toast("⚠️ Already in wishlist!", icon="ℹ️")
                return

        # Add new
        sheet.append_row([username, title, datetime.now().strftime("%Y-%m-%d")])
        st.toast("✅ Added to Wishlist!", icon="🎉")
        
    except Exception as e:
        st.error(f"Save failed: {str(e)}")

def remove_from_wishlist(username: str, title: str) -> bool:
    try:
        sheet = get_sheet("Wishlist")
        data = sheet.get_all_values()
        for i, row in enumerate(data):
            if len(row) > 1 and row[0] == username and row[1] == title:
                sheet.delete_rows(i + 1)
                return True
        return False
    except: return False

# ==========================================
# ⭐ REVIEWS
# ==========================================
def add_review(title: str, username: str, rating: int, comment: str) -> bool:
    try:
        sheet = get_sheet("Reviews")
        sheet.append_row([title, username, rating, comment, datetime.now().strftime("%Y-%m-%d")])
        return True
    except: return False

def get_reviews(title: str) -> List[Dict]:
    try:
        data = get_sheet("Reviews").get_all_values()[1:]
        return [{"user": r[1], "rating": int(r[2]), "comment": r[3], "date": r[4]} 
                for r in data if len(r) > 3 and r[0] == title]
    except: return []

def get_avg_rating(title: str) -> Tuple[float, int]:
    reviews = get_reviews(title)
    if not reviews: return 0.0, 0
    ratings = [r["rating"] for r in reviews]
    return sum(ratings) / len(ratings), len(ratings)

# ==========================================
# 🎨 UI COMPONENTS
# ==========================================
def render_book_card(book: Dict, username: str):
    col1, col2 = st.columns([1, 3])
    with col1:
        if book["image"]: st.image(book["image"], use_container_width=True)
        else: st.info("📖")
    
    with col2:
        st.markdown(f"### {book['title']}")
        st.caption(f"by {book['authors']}")
        
        if book["rating"] > 0:
            st.caption(f"{'⭐' * int(book['rating'])} ({book['rating']}/5) - {book['rating_count']} reviews")
        
        st.write(f"**Publisher:** {book['publisher']} | **Published:** {book['date']}")
        
        col_a, col_b, col_c = st.columns(3)
        
        # 1. Wishlist Button
        with col_a:
            if st.button("❤️ Wishlist", key=f"w_{book['title']}_{datetime.now().microsecond}"):
                add_to_wishlist(username, book['title'])
                # Trick: We don't need explicit rerun here because button click triggers script rerun automatically
        
        # 2. Preview Link
        with col_b:
            if book["link"]: st.link_button("🔗 Preview", book["link"])
        
        # 3. Review Button
        with col_c:
            if st.button("⭐ Review", key=f"r_{book['title']}"):
                st.session_state[f"review_{book['title']}"] = True

def render_review_form(title: str, username: str):
    if st.session_state.get(f"review_{title}", False):
        with st.expander(f"✍️ Review: {title}", expanded=True):
            with st.form(key=f"form_{title}"):
                rating = st.slider("Rating", 1, 5, 5)
                comment = st.text_area("Your thoughts...")
                if st.form_submit_button("Submit"):
                    if add_review(title, username, rating, comment):
                        st.success("✅ Review posted!")
                        st.session_state[f"review_{title}"] = False
                        st.rerun()
    
    reviews = get_reviews(title)
    if reviews:
        with st.expander(f"📖 {len(reviews)} Reviews"):
            for r in reviews[-5:]:
                st.markdown(f"**{r['user']}** {'⭐' * r['rating']}\n\n{r['comment']}\n\n*{r['date']}*")
                st.divider()

# ==========================================
# 🏠 MAIN APP
# ==========================================
def main_app():
    username = st.session_state["username"]
    
    # Init AI
    if "ai" not in st.session_state: st.session_state.ai = AIHelper()
    
    # Sidebar
    with st.sidebar:
        st.title(f"👤 {username}")
        stats = get_user_stats(username)
        c1, c2 = st.columns(2)
        c1.metric("Wishlist", stats["wishlist"])
        c2.metric("Reviews", stats["reviews"])
        
        st.divider()
        st.subheader("❤️ My Wishlist")
        wishlist = get_wishlist(username)
        if wishlist:
            for item in wishlist[-5:]:
                st.caption(f"📖 {item['title']}")
        else:
            st.info("Empty")
        
        st.divider()
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.clear()
            st.rerun()
    
    # Main Content
    st.title("📚 BookBot Pro")
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Discover", "💡 Recommendations", "📖 Library", "📄 PDF Study"])
    
    # TAB 1: DISCOVER
    with tab1:
        query = st.text_input("Search for books...", placeholder="Enter title or author")
        if st.button("🔎 Search", use_container_width=True):
            if query:
                with st.spinner("Searching..."):
                    results = search_books(query)
                if results:
                    for book in results:
                        render_book_card(book, username)
                        with st.expander("🤖 AI Summary", expanded=True):
                            st.write(st.session_state.ai.summarize_book(book))
                        render_review_form(book['title'], username)
                else:
                    st.error(f"❌ No books found for '{query}'")

    # TAB 2: RECOMMENDATIONS
    with tab2:
        prefs = st.text_area("What do you like?")
        if st.button("✨ Get Recommendations", use_container_width=True) and prefs:
            with st.spinner("Thinking..."):
                st.markdown(st.session_state.ai.recommend_books(prefs))

    # TAB 3: LIBRARY (Full Wishlist Management)
    with tab3:
        st.header(f"📖 {username}'s Library")
        wishlist = get_wishlist(username)
        if wishlist:
            df = pd.DataFrame(wishlist)
            df.index = range(1, len(df) + 1)
            st.dataframe(df, use_container_width=True)
            
            to_remove = st.selectbox("Select to remove:", [w['title'] for w in wishlist])
            if st.button("Remove Book", use_container_width=True):
                if remove_from_wishlist(username, to_remove):
                    st.success(f"✅ Removed '{to_remove}'")
                    st.rerun()
        else:
            st.info("📚 Your library is empty!")

    # TAB 4: PDF
    with tab4:
        pdf = st.file_uploader("Upload PDF", type="pdf")
        if pdf:
            if st.session_state.get("pdf_name") != pdf.name:
                with st.spinner("Processing..."):
                    db = process_pdf(pdf)
                    if db:
                        st.session_state.pdf_db = db
                        st.session_state.pdf_name = pdf.name
                        st.success("✅ PDF loaded!")
            
            if "pdf_db" in st.session_state:
                q = st.text_input("Ask PDF:")
                if st.button("Analyze", use_container_width=True) and q:
                    docs = st.session_state.pdf_db.similarity_search(q, k=3)
                    ctx = "\n".join([d.page_content for d in docs])
                    st.write(st.session_state.ai.answer_pdf(ctx, q))

# ==========================================
# 🔐 LOGIN
# ==========================================
if "username" not in st.session_state:
    st.title("🔐 BookBot Login")
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        with st.form("login"):
            st.info("New users auto-registered")
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.form_submit_button("Enter", use_container_width=True):
                if u and p:
                    status, msg = login_user(u, p)
                    if status in ["success", "register"]:
                        st.session_state["username"] = u
                        st.success(msg)
                        st.rerun()
                    else: st.error(msg)
else:
    main_app()
