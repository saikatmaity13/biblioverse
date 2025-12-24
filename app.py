import os
import ssl
import pandas as pd
import requests
import streamlit as st
import pypdf
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from huggingface_hub import InferenceClient

# ==========================================
# 🔧 SETUP
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

# 🔑 API TOKEN
if "HUGGINGFACE_TOKEN" in st.secrets:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACE_TOKEN"]
else:
    # If running locally, you can load it from an environment variable or a local file
    # BUT for now, to bypass the git error, we won't hardcode it here.
    st.error("⚠️ HuggingFace Token not found! Please set it in Streamlit Secrets.")
    st.stop()

# ==========================================
# ☁️ GOOGLE SHEETS STORAGE (PERMANENT)
# ==========================================

@st.cache_resource
def get_google_sheet():
    """Connects to Google Sheets using secrets."""
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    
    # OPTION A: If running locally with a file
    if os.path.exists("secrets.json"):
        creds = ServiceAccountCredentials.from_json_keyfile_name("secrets.json", scope)
    
    # OPTION B: If deployed on Streamlit Cloud (Secrets Management)
    elif "gcp_service_account" in st.secrets:
        creds = ServiceAccountCredentials.from_json_keyfile_dict(st.secrets["gcp_service_account"], scope)
    else:
        st.error("⚠️ Missing 'secrets.json' or Streamlit Secrets!")
        st.stop()
        
    client = gspread.authorize(creds)
    # Open the sheet. Make sure you created a sheet named "BookBot_Data"
    return client.open("BookBot_Data").sheet1

# --- 1. Wishlist Functions (Cloud) ---
def get_saved_books():
    try:
        sheet = get_google_sheet()
        # Column 1 is for Wishlist
        return [b for b in sheet.col_values(1) if b != "Wishlist_Books"] 
    except:
        return []

def save_book_callback(book_title):
    try:
        sheet = get_google_sheet()
        existing = sheet.col_values(1)
        if book_title in existing:
            st.toast(" Already saved!", icon="ℹ️")
            return
        
        # Write to Column A (Row 1 is header, so append)
        # Note: We are using a simple append, but mixing columns can be tricky.
        # Ideally, use separate sheets. For now, we assume Col A=Wishlist.
        sheet.update_cell(len(existing) + 1, 1, book_title)
        st.toast(f" Saved '{book_title}' to Cloud!", icon="🎉")
    except Exception as e:
        st.error(f"Cloud Error: {e}")

def remove_book_callback(book_title):
    try:
        sheet = get_google_sheet()
        cell = sheet.find(book_title)
        if cell:
            sheet.update_cell(cell.row, cell.col, "") # Clear it
            st.toast(" Removed from Cloud.")
    except Exception as e:
        st.error(f"Remove Failed: {e}")

# --- 2. Reviews Functions (Cloud) ---
# We will use Columns C, D, E, F, G for Reviews
# C=Title, D=User, E=Rating, F=Comment, G=Date

def get_reviews_for_book(book_title):
    try:
        sheet = get_google_sheet()
        data = sheet.get_all_values() # Read everything
        reviews = []
        # Skip header if exists
        for row in data:
            # Check if row has enough columns (at least 7)
            if len(row) > 3 and row[2] == book_title: # Col C is index 2
                reviews.append({
                    "User": row[3],      # Col D
                    "Rating": row[4],    # Col E
                    "Comment": row[5],   # Col F
                    "Date": row[6]       # Col G
                })
        return reviews
    except:
        return []

def add_review(book_title, user, rating, comment):
    try:
        sheet = get_google_sheet()
        # Append row to the end of the sheet
        # Structure: [Wishlist(skip), skip, Title, User, Rating, Comment, Date]
        row_data = ["", "", book_title, user, rating, comment, datetime.now().strftime("%Y-%m-%d")]
        sheet.append_row(row_data)
        return True
    except Exception as e:
        st.error(f"Review Error: {e}")
        return False

# ==========================================
# 🤖 AI BRAIN
# ==========================================
@st.cache_resource
def setup_brain():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(persist_directory="./library_db", embedding_function=embeddings)
    client = InferenceClient(model="Qwen/Qwen2.5-7B-Instruct", token=os.environ["HUGGINGFACEHUB_API_TOKEN"])
    return db, client, embeddings

try:
    vector_db, client, embeddings = setup_brain()
except Exception as e:
    st.error(f"Setup Error: {e}")
    st.stop()

def process_pdf(uploaded_file):
    reader = pypdf.PdfReader(uploaded_file)
    text = "".join([page.extract_text() or "" for page in reader.pages])
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).create_documents([text])
    return Chroma.from_documents(chunks, embeddings)

# ==========================================
# 📂 SIDEBAR
# ==========================================
with st.sidebar:
    st.title(" Library Card")
    st.divider()
    st.header("  Wishlist")
    
    saved_list = get_saved_books()
    if saved_list:
        for book in saved_list:
            if book.strip(): st.write(f" {book}")
    else:
        st.info("No books saved.")

# ==========================================
# 🕵️ LOGIC
# ==========================================

def search_google_books(query):
    url = "https://www.googleapis.com/books/v1/volumes"
    try:
        res = requests.get(url, params={"q": query, "maxResults": 1, "langRestrict": "en"}).json()
        if "items" in res:
            book = res["items"][0]["volumeInfo"]
            rating = book.get("averageRating", None)
            star_str = ("⭐" * int(rating) + ("½" if rating % 1 >= 0.5 else "")) if rating else ""
            
            return {
                "text": f"Title: {book.get('title')}\nAuthor: {', '.join(book.get('authors', []))}\nDesc: {book.get('description')}",
                "title": book.get('title'), 
                "image": book.get('imageLinks', {}).get('thumbnail'),
                "rating": rating, "count": book.get("ratingsCount", 0), "stars": star_str,
                "link": book.get("previewLink", book.get("infoLink", None)),
                "found": True
            }
    except: pass
    return {"found": False}

def ask_ai(sys, msg):
    try:
        return client.chat_completion([{"role": "system", "content": sys}, {"role": "user", "content": msg}], max_tokens=800).choices[0].message.content
    except: return "AI Error"

def process_chat(query):
    scout = search_google_books(query)
    sys = f"Librarian AI. Answer based on: {scout['text']}" if scout["found"] else "Librarian AI."
    if scout["found"]: 
        vector_db.add_documents([Document(page_content=scout["text"], metadata={"title": scout["title"]})])
    return ask_ai(sys, query), scout

def get_recs(taste): return ask_ai("Recommend 3 books.", f"I love: {taste}")

# ==========================================
# 🖥️ UI
# ==========================================
st.title("Biblioverse")

tab1, tab2, tab3, tab4 = st.tabs([" Chat", " Recs", "Wishlist", " Study Room"])

with tab1:
    if "messages" not in st.session_state: st.session_state.messages = []
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if "image" in msg: st.image(msg["image"], width=130)

    if prompt := st.chat_input("Enter book name..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.spinner("Analyzing..."):
            ans, scout = process_chat(prompt)
            st.chat_message("assistant").write(ans)
            
            if scout["found"]:
                col1, col2 = st.columns([1, 3])
                with col1:
                    if scout["image"]: st.image(scout["image"], width=130)
                with col2:
                    if scout["stars"]: st.caption(f"{scout['stars']} ({scout['count']} reviews)")
                    
                    c1, c2 = st.columns(2)
                    with c1: st.button(f" Save Cloud", key=f"save_{scout['title']}", on_click=save_book_callback, args=(scout["title"],))
                    with c2: 
                        if scout["link"]: st.link_button(" Google Reviews", scout["link"])
                
                # REVIEWS SECTION
                st.divider()
                st.subheader("  Reviews")
                my_reviews = get_reviews_for_book(scout["title"])
                
                if my_reviews:
                    for r in my_reviews:
                        with st.chat_message("user", avatar="👤"):
                            st.write(f"**{r['User']}**: {r['Comment']} ({'⭐'*int(r['Rating'])})")
                else:
                    st.info("No reviews yet.")

                with st.expander(" Add Review"):
                    with st.form("new_review"):
                        u = st.text_input("Name")
                        r = st.slider("Rating", 1, 5, 5)
                        c = st.text_area("Comment")
                        if st.form_submit_button("Post"):
                            if add_review(scout["title"], u, r, c):
                                st.success("Saved to Cloud!")
                                st.rerun()

            st.session_state.messages.append({"role": "assistant", "content": ans, "image": scout["image"]})

with tab2:
    t = st.text_area("I like...")
    if st.button("Get Recs"): st.markdown(get_recs(t))

with tab3:
    st.header("  Wishlist")
    books = get_saved_books()
    if books:
        st.dataframe(pd.DataFrame(books, columns=["Title"]), use_container_width=True)
        b = st.selectbox("Remove:", books)
        st.button("Delete", on_click=remove_book_callback, args=(b,))

with tab4:
    st.header(" Study Room")
    f = st.file_uploader("Upload PDF", type="pdf")
    if f:
        if "pdf_name" not in st.session_state or st.session_state.pdf_name != f.name:
            st.session_state.db = process_pdf(f)
            st.session_state.pdf_name = f.name
            st.success("PDF Loaded!")
        
        q = st.text_input("Ask PDF:")
        if st.button("Ask"):
            docs = st.session_state.db.similarity_search(q, k=3)
            ctx = "\n".join([d.page_content for d in docs])

            st.write(ask_ai(f"Context: {ctx}", q))

