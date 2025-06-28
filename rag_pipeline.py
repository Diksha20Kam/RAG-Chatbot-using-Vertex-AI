import os
import requests
import time
import pdfplumber
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS

# Load API key
load_dotenv()
API_KEY = os.getenv("VERTEX_API_KEY")

EMBEDDING_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/textembedding-gecko:embedContent?key={API_KEY}"
GEMINI_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={API_KEY}"
HEADERS = {"Content-Type": "application/json"}

# 1. Load and split PDF
def load_and_split_pdf(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            texts = [page.extract_text() for page in pdf.pages if page.extract_text()]
        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        return splitter.create_documents(texts)
    except Exception as e:
        raise RuntimeError(f"PDF load failed: {e}")

# 2. Get embeddings
def get_embeddings(docs):
    embeddings = []
    for doc in docs:
        body = {"content": doc.page_content}
        try:
            res = requests.post(EMBEDDING_ENDPOINT, headers=HEADERS, json=body)
            res.raise_for_status()
            json_res = res.json()
            embeddings.append(json_res.get("embedding", [0.0] * 768))
        except Exception as e:
            print(f"❌ Embedding failed: {e}")
            embeddings.append([0.0] * 768)
        time.sleep(1)  # prevent rate limiting
    return embeddings

# 3. Build FAISS vector store
def build_vector_store(docs):
    embeddings = get_embeddings(docs)
    return FAISS.from_embeddings(list(zip(embeddings, docs)))

# 4. Retrieve top K docs
def retrieve_context(query, faiss_store):
    body = {"content": query}
    res = requests.post(EMBEDDING_ENDPOINT, headers=HEADERS, json=body)
    res.raise_for_status()
    embedding = res.json()["embedding"]
    return faiss_store.similarity_search_by_vector(embedding, k=3)

# 5. Ask Gemini
def ask_gemini(context, query):
    prompt = f"Use the context below to answer the question.\n\nContext:\n{context}\n\nQuestion: {query}"
    body = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    res = requests.post(GEMINI_ENDPOINT, headers=HEADERS, json=body)
    try:
        return res.json()['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        return f"❌ Error generating response: {e}"
