import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
import faiss
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv


os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not available in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                                   temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(user_question, pdf_docs):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # 1. Retrieve the vector dimension (replace with the correct method if needed)
    vector_dimension = embeddings.get_vector_dimension()  # Assuming this method exists

    # 2. Create the FAISS vector store using the dimension
    vector_store = faiss.IndexFlatL2(vector_dimension)

    # Process uploaded PDFs and extract text
    raw_text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(raw_text)

    # Add embeddings to the vector store
    for chunk in text_chunks:
        embeddings = embeddings.encode(chunk)
        vector_store.add(embeddings)

    # Use vector_store for similarity search and chain processing
    docs = vector_store.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])



def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini")

    user_question = st.text_input("Ask a Question from the PDF Files")
    pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)

    if user_question and pdf_docs:
        raw_text = get_pdf_text(pdf_docs)
        text_chunks = get_text_chunks(raw_text)
        user_input(user_question, text_chunks)

    with st.sidebar:
        st.title("Menu:")


if __name__ == "__main__":
    main()
