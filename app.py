import streamlit as st
from huggingface_hub import login
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Streamlit UI setup
st.title("PDF Question Answering System")
st.write("Upload a PDF document and ask questions!")

# User inputs Hugging Face token and PDF file
hf_token = st.text_input("Enter your Hugging Face token:", type="password", value="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")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file and hf_token:
    # Authenticate with Hugging Face token
    login(hf_token)

    # Load the PDF file
    with open("uploaded_pdf.pdf", "wb") as f:
        f.write(uploaded_file.read())

    loader = PyMuPDFLoader("uploaded_pdf.pdf")
    documents = loader.load()

    # Split the documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)

    # Load embeddings using the model directly from Hugging Face
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},  # Change 'cpu' to 'cuda' if you have a GPU
        encode_kwargs={'normalize_embeddings': False}
    )

    # Create a FAISS vector store with the documents
    db = FAISS.from_documents(docs, embeddings)

    # Load the T5 model for text generation directly from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)

    # Create an LLM pipeline
    llm = HuggingFacePipeline(
        pipeline=pipe,
        model_kwargs={"temperature": 0},
    )

    # Define the QA prompt template
    template = """You are a Question-Answering Expert Bot. Your job is to answer the query given by the user.
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    # Set up the QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

    # User input for the question
    question = st.text_input("Enter your question:")
    if st.button("Get Answer") and question:
        # Perform the query and get the result
        result = qa_chain.run(question)
        st.write("**Answer:**")
        st.write(result)
