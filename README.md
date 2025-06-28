# 🧠 RAG Chatbot using Vertex AI + LangChain

This project is a **Retrieval-Augmented Generation (RAG) chatbot** that allows users to ask questions about a long PDF document (100–200 pages). The chatbot uses **Vertex AI (Google)** for embedding and generation, **LangChain** for chunking and retrieval, and **Streamlit** for the user interface.

---

## 🚀 Features

- 🔍 PDF ingestion using `pdfplumber`
- 🧩 Text splitting into chunks using LangChain
- 🔎 FAISS-based retrieval of relevant chunks
- 🧠 Vertex AI:
  - `textembedding-gecko` for embedding
  - `gemini-pro` for content generation
- 💬 Streamlit UI

---

## 📁 Folder Structure

```
rag_chatbot_project/
│
├── app.py                # Streamlit UI
├── rag_pipeline.py       # Core RAG logic
├── requirements.txt      # Dependencies
├── .env.example          # API key template
└── README.md             # Project overview
```

---

## ⚙️ Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/Diksha20Kam/RAG-Chatbot-using-Vertex-AI.git
cd RAG-Chatbot-using-Vertex-AI
```

### 2. Create & activate conda env

```bash
conda create -n rag_chatbot_env python=3.10 -y
conda activate rag_chatbot_env
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your API key

Create a `.env` file in the root directory:

```
VERTEX_API_KEY=your_actual_vertex_api_key
```

### 5. Run Streamlit app

```bash
streamlit run app.py
```

---

## 🛠️ Requirements

```
streamlit
langchain
langchain-community
python-dotenv
pdfplumber
faiss-cpu
requests
```

---

