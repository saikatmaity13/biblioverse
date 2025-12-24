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
# ☁️ DATABASE MANAGER
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
    try:
        return get_sheets_client().open("BookBot_Data").worksheet(name)
    except:
        # Auto-create
        sh = get_sheets_client().open("BookBot_Data")
        ws = sh.add_worksheet(title=name, rows="1000", cols="5")
        headers = {
            "Reviews": ["Book Title", "Username", "Rating", "Comment", "Date"],
            "Users": ["Username", "Password", "JoinDate"]
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
        
        # Register new user
        sheet.append_row([username, hash_password(password), datetime.now().strftime("%Y-%m-%d")])
        return "register", "Account created!"
    except Exception as e:
        return "error", str(e)

def get_user_stats(username: str) -> Dict:
    try:
        wishlist_data = get_sheet("Sheet1").get_all_values()
        review_data = get_sheet("Reviews").get_all_values()
        
        w_count = len([r for r in wishlist_data if len(r) > 1 and r[0] == username])
        r_count = len([r for r in review_data if len(r) > 1 and r[1] == username])
        
        return {"wishlist": w_count, "reviews": r_count}
    except:
        return {"wishlist": 0, "reviews": 0}

# ==========================================
# 📚 BOOK SEARCH
# ==========================================
def search_books(query: str, max_results: int = 5) -> List[Dict]:
    """Search Google Books API with multiple fallback strategies"""
    
    if not query or len(query.strip()) == 0:
        return []
    
    base_url = "https://www.googleapis.com/books/v1/volumes"
    
    # Clean query
    clean_query = query.strip()
    
    # Multiple search attempts
    search_params_list = [
        {"q": clean_query, "maxResults": max_results},
        {"q": clean_query, "maxResults": max_results, "printType": "books"},
        {"q": f"inauthor:{clean_query}", "maxResults": max_results},
        {"q": f"intitle:{clean_query}", "maxResults": max_results},
    ]
    
    all_books = []
    
    for params in search_params_list:
        try:
            # Make request with longer timeout
            response = requests.get(
                base_url, 
                params=params, 
                timeout=15,
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            
            # Check if request was successful
            if response.status_code != 200:
                print(f"API returned status {response.status_code}")
                continue
            
            data = response.json()
            
            # Check for errors in response
            if "error" in data:
                print(f"API error: {data['error']}")
                continue
            
            # Check if we have items
            if "items" not in data or len(data["items"]) == 0:
                continue
            
            # Parse books
            for item in data["items"]:
                try:
                    info = item.get("volumeInfo", {})
                    
                    if not info.get("title"):
                        continue
                    
                    # Get image links
                    img_links = info.get("imageLinks", {})
                    image_url = (
                        img_links.get("thumbnail") or 
                        img_links.get("smallThumbnail") or 
                        img_links.get("small") or
                        None
                    )
                    
                    # Build book object
                    book = {
                        "title": info.get("title", "Unknown Title"),
                        "authors": ", ".join(info.get("authors", ["Unknown Author"])),
                        "description": info.get("description", "No description available."),
                        "image": image_url,
                        "rating": float(info.get("averageRating", 0)),
                        "rating_count": int(info.get("ratingsCount", 0)),
                        "link": info.get("previewLink") or info.get("infoLink") or "#",
                        "publisher": info.get("publisher", "Unknown"),
                        "date": info.get("publishedDate", "Unknown"),
                        "pages": int(info.get("pageCount", 0)),
                        "categories": ", ".join(info.get("categories", ["Uncategorized"]))
                    }
                    
                    # Avoid duplicates
                    if not any(b["title"] == book["title"] for b in all_books):
                        all_books.append(book)
                    
                    if len(all_books) >= max_results:
                        return all_books[:max_results]
                
                except Exception as e:
                    print(f"Error parsing book: {e}")
                    continue
            
            # If we found books in this attempt, return them
            if all_books:
                return all_books[:max_results]
        
        except requests.exceptions.Timeout:
            print(f"Request timeout for params: {params}")
            continue
        except requests.exceptions.ConnectionError:
            print("Connection error - check internet")
            continue
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            continue
        except Exception as e:
            print(f"Unexpected error: {e}")
            continue
    
    return all_books

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
            res = self.client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                max_tokens=200
            )
            return res.choices[0].message.content
        except:
            return "Summary unavailable"
    
    def recommend_books(self, preferences: str) -> str:
        try:
            res = self.client.chat_completion(
                messages=[{"role": "user", "content": f"Recommend 5 books for: {preferences}"}],
                model=self.model,
                max_tokens=500
            )
            return res.choices[0].message.content
        except:
            return "Recommendations unavailable"
    
    def answer_pdf(self, context: str, question: str) -> str:
        try:
            res = self.client.chat_completion(
                messages=[
                    {"role": "system", "content": f"Answer based on:\n{context}"},
                    {"role": "user", "content": question}
                ],
                model=self.model,
                max_tokens=400
            )
            return res.choices[0].message.content
        except:
            return "Error processing question"

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
        
        if len(text.strip()) < 100:
            return None
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.create_documents([text])
        return Chroma.from_documents(chunks, get_embeddings())
    except:
        return None

# ==========================================
# 📖 WISHLIST
# ==========================================
def get_wishlist(username: str) -> List[Dict]:
    try:
        data = get_sheet("Sheet1").get_all_values()[1:]
        return [{"title": r[1], "date": r[2] if len(r) > 2 else "Unknown"} 
                for r in data if len(r) > 1 and r[0] == username]
    except:
        return []

def add_to_wishlist(username: str, title: str) -> Tuple[bool, str]:
    try:
        sheet = get_sheet("Sheet1")
        existing = sheet.get_all_values()
        
        for row in existing:
            if len(row) > 1 and row[0] == username and row[1] == title:
                return False, "Already in wishlist"
        
        sheet.append_row([username, title, datetime.now().strftime("%Y-%m-%d")])
        return True, "Added to wishlist!"
    except:
        return False, "Error adding"

def remove_from_wishlist(username: str, title: str) -> bool:
    try:
        sheet = get_sheet("Sheet1")
        data = sheet.get_all_values()
        for i, row in enumerate(data):
            if len(row) > 1 and row[0] == username and row[1] == title:
                sheet.delete_rows(i + 1)
                return True
        return False
    except:
        return False

# ==========================================
# ⭐ REVIEWS
# ==========================================
def add_review(title: str, username: str, rating: int, comment: str) -> bool:
    try:
        sheet = get_sheet("Reviews")
        sheet.append_row([title, username, rating, comment, datetime.now().strftime("%Y-%m-%d")])
        return True
    except:
        return False

def get_reviews(title: str) -> List[Dict]:
    try:
        data = get_sheet("Reviews").get_all_values()[1:]
        return [{"user": r[1], "rating": int(r[2]), "comment": r[3], "date": r[4]} 
                for r in data if len(r) > 3 and r[0] == title]
    except:
        return []

def get_avg_rating(title: str) -> Tuple[float, int]:
    reviews = get_reviews(title)
    if not reviews:
        return 0.0, 0
    ratings = [r["rating"] for r in reviews]
    return sum(ratings) / len(ratings), len(ratings)

# ==========================================
# 🎨 UI COMPONENTS
# ==========================================
def render_book_card(book: Dict, username: str):
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if book["image"]:
            st.image(book["image"], use_container_width=True)
        else:
            st.info("📖")
    
    with col2:
        st.markdown(f"### {book['title']}")
        st.caption(f"by {book['authors']}")
        
        if book["rating"] > 0:
            st.caption(f"{'⭐' * int(book['rating'])} ({book['rating']}/5) - {book['rating_count']} reviews")
        
        avg_rating, count = get_avg_rating(book['title'])
        if count > 0:
            st.caption(f"📱 BookBot: {'⭐' * int(avg_rating)} ({avg_rating:.1f}/5) - {count} reviews")
        
        st.write(f"**Publisher:** {book['publisher']} | **Pages:** {book['pages']} | **Published:** {book['date']}")
        
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            if st.button("❤️ Wishlist", key=f"wish_{book['title']}_{id(book)}"):
                success, msg = add_to_wishlist(username, book['title'])
                st.toast(msg, icon="✅" if success else "ℹ️")
        
        with col_b:
            if book["link"]:
                st.link_button("🔗 Preview", book["link"])
        
        with col_c:
            if st.button("⭐ Review", key=f"rev_{book['title']}_{id(book)}"):
                st.session_state[f"review_{book['title']}"] = True

def render_review_form(title: str, username: str):
    if st.session_state.get(f"review_{title}", False):
        with st.expander(f"✍️ Review: {title}", expanded=True):
            with st.form(key=f"form_{title}"):
                rating = st.slider("Rating", 1, 5, 5)
                comment = st.text_area("Your thoughts...")
                
                col1, col2 = st.columns(2)
                with col1:
                    submit = st.form_submit_button("Submit")
                with col2:
                    cancel = st.form_submit_button("Cancel")
                
                if submit and comment:
                    if add_review(title, username, rating, comment):
                        st.success("✅ Review posted!")
                        st.session_state[f"review_{title}"] = False
                        st.rerun()
                
                if cancel:
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
    if "ai" not in st.session_state:
        st.session_state.ai = AIHelper()
    
    # Sidebar
    with st.sidebar:
        st.title(f"👤 {username}")
        
        stats = get_user_stats(username)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📚 Wishlist", stats["wishlist"])
        with col2:
            st.metric("⭐ Reviews", stats["reviews"])
        
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
    
    # Main UI
    st.title("📚 BookBot Pro")
    st.caption("Your AI-Powered Reading Companion")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Discover", "💡 Recommendations", "📖 Library", "📄 PDF Study"])
    
    # TAB 1: DISCOVER
    with tab1:
        st.header("🔍 Discover Books")
        
        query = st.text_input("Search for books...", placeholder="Enter title or author")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            search = st.button("🔎 Search", use_container_width=True)
        with col2:
            max_res = st.selectbox("Results", [3, 5, 10], index=1)
        
        if search and query:
            with st.spinner("🔍 Searching Google Books..."):
                results = search_books(query, max_res)
            
            if results:
                st.success(f"✅ Found {len(results)} books!")
                
                for book in results:
                    render_book_card(book, username)
                    
                    with st.expander("🤖 AI Summary"):
                        summary = st.session_state.ai.summarize_book(book)
                        st.write(summary)
                    
                    render_review_form(book['title'], username)
                    st.divider()
            else:
                st.error(f"❌ No books found for '{query}'")
                st.info("💡 **Possible issues:**\n- Google Books API might be blocked\n- Check your internet connection\n- Try a different search term")
                
                # Manual test
                with st.expander("🔧 Test API Connection"):
                    if st.button("Test Connection"):
                        test_url = "https://www.googleapis.com/books/v1/volumes?q=harry+potter&maxResults=1"
                        try:
                            test_response = requests.get(test_url, timeout=10)
                            st.write(f"**Status Code:** {test_response.status_code}")
                            
                            if test_response.status_code == 200:
                                test_data = test_response.json()
                                if "items" in test_data:
                                    st.success("✅ API is working! Try a different book title.")
                                    st.json(test_data["items"][0]["volumeInfo"])
                                else:
                                    st.warning("API responded but no items found")
                                    st.json(test_data)
                            else:
                                st.error(f"API returned error code: {test_response.status_code}")
                                st.text(test_response.text)
                        except Exception as e:
                            st.error(f"❌ Connection Error: {str(e)}")
                            st.info("The Google Books API might be blocked by your network/firewall.")
    
    # TAB 2: RECOMMENDATIONS
    with tab2:
        st.header("💡 Get Recommendations")
        
        prefs = st.text_area(
            "What do you like?",
            placeholder="e.g., Mystery novels, sci-fi, historical fiction...",
            height=100
        )
        
        if st.button("✨ Get Recommendations", use_container_width=True):
            if prefs:
                with st.spinner("Thinking..."):
                    recs = st.session_state.ai.recommend_books(prefs)
                st.markdown("### 📚 Recommended:")
                st.markdown(recs)
    
    # TAB 3: LIBRARY
    with tab3:
        st.header(f"📖 {username}'s Library")
        
        wishlist = get_wishlist(username)
        
        if wishlist:
            df = pd.DataFrame(wishlist)
            df.index = range(1, len(df) + 1)
            st.dataframe(df, use_container_width=True)
            
            st.subheader("🗑️ Remove Books")
            to_remove = st.selectbox("Select:", [w['title'] for w in wishlist])
            
            if st.button("Remove", use_container_width=True):
                if remove_from_wishlist(username, to_remove):
                    st.success(f"✅ Removed '{to_remove}'")
                    st.rerun()
        else:
            st.info("📚 Your library is empty!")
    
    # TAB 4: PDF STUDY
    with tab4:
        st.header("📄 PDF Study Room")
        
        pdf = st.file_uploader("Upload PDF", type="pdf")
        
        if pdf:
            if st.session_state.get("pdf_name") != pdf.name:
                with st.spinner("Processing PDF..."):
                    db = process_pdf(pdf)
                    if db:
                        st.session_state.pdf_db = db
                        st.session_state.pdf_name = pdf.name
                        st.success("✅ PDF loaded!")
                    else:
                        st.error("❌ Could not process PDF")
            
            if "pdf_db" in st.session_state:
                question = st.text_input("Ask a question about the PDF:")
                
                if st.button("🔍 Analyze", use_container_width=True):
                    if question:
                        docs = st.session_state.pdf_db.similarity_search(question, k=3)
                        context = "\n".join([d.page_content for d in docs])
                        
                        with st.spinner("Analyzing..."):
                            answer = st.session_state.ai.answer_pdf(context, question)
                        
                        st.markdown("### 💡 Answer:")
                        st.write(answer)

# ==========================================
# 🔐 LOGIN
# ==========================================
if "username" not in st.session_state:
    st.title("🔐 BookBot Login")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
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
                    else:
                        st.error(msg)
else:
    main_app()
