import os
import streamlit as st
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
print("API Key Loaded:", api_key is not None)

st.set_page_config(page_title="AI STEM Tutor", layout="centered")
st.title(" AI Tutor for STEM Subjects")
st.markdown("Powered by Agentic AI, RAG, and Granite Principles")

# Upload PDF
uploaded_file = st.file_uploader("Upload a STEM textbook or PDF", type="pdf")
if uploaded_file:
    with open("chapter.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Load and split document
    loader = PyPDFLoader("chapter.pdf")
    docs = loader.load()

    # Embed and store
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)
    retriever = db.as_retriever()

    # RAG + Agentic AI
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0),
        chain_type="stuff",
        retriever=retriever
    )

    st.success("PDF processed! Ask your question below ")
    question = st.text_input("Ask a STEM Question (e.g., Explain Ohm’s Law)")
    
    if question:
        ethical_prompt = (
            "As an ethical AI STEM tutor, explain this using only the uploaded content. "
            "Be step-by-step, simple, and avoid hallucination.\n"
            f"Question: {question}"
        )
        response = qa_chain.run(ethical_prompt)
        st.write("### Answer:")
        st.success(response)




