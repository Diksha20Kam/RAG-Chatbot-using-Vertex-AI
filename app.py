import streamlit as st
from rag_pipeline import (
    load_and_split_pdf,
    build_vector_store,
    retrieve_context,
    ask_gemini
)

st.set_page_config(page_title="🧠 RAG Chatbot with Vertex AI")

st.title("📄 RAG Chatbot using Vertex AI + LangChain")
st.markdown("Ask questions based on the PDF.")

pdf_path = "/Users/admin/Downloads/Mastering RAG.pdf"

if "faiss_store" not in st.session_state:
    with st.spinner("🔍 Reading and embedding PDF..."):
        try:
            docs = load_and_split_pdf(pdf_path)
            st.session_state.faiss_store = build_vector_store(docs)
            st.success("✅ PDF processed successfully!")
        except Exception as e:
            st.error(f"Error loading PDF: {e}")

query = st.text_input("Ask your question:")

if query and "faiss_store" in st.session_state:
    with st.spinner("💬 Generating answer..."):
        context_docs = retrieve_context(query, st.session_state.faiss_store)
        combined_context = "\n\n".join([doc.page_content for doc in context_docs])
        answer = ask_gemini(combined_context, query)
        st.markdown("### 🤖 Answer:")
        st.write(answer)