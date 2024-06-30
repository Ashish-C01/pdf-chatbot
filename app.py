import streamlit as st
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
import fitz
from dotenv import load_dotenv
import os
load_dotenv()

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
st.title('Document Q&A')
data_uploaded=False

def get_chain():
    model = ChatGoogleGenerativeAI(model='gemini-1.5-pro-latest', temperature=0.1)
    prompt_ = """
    Answer the questions as detailed as possible from the provided context, make sure to provide all the 
        details, if the answer is not in the provided context just say, "answer is not available in context",   
        don't provide the wrong answer\n. 
        context: {context}
        Questions:{question}
        Answer:
    """
    prompt = PromptTemplate(template=prompt_, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def get_pdf_content(pdffile):
    with fitz.open(stream=pdffile.read(), filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text


def create_database(data):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_document = text_splitter.split_text(data)
    vectors = FAISS.from_texts(final_document, embeddings)
    vectors.save_local("faiss_index")


def user_input(u_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local('faiss_index', embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(u_question)
    chain = get_chain()
    response = chain(
        {"input_documents": docs, "question": u_question}, return_only_outputs=True
    )
    return response["output_text"]


with st.sidebar:
    uploaded_file = st.file_uploader("Upload pdf file", key="pdf_uploader")
    if st.button('Create vector store'):
        if uploaded_file is not None:
            data = get_pdf_content(uploaded_file)
            create_database(data)
            st.write("Vector store created")
        else:
            st.write("Please upload pdf file")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("Ask questions"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner(text="Fetching details..."):
            response = user_input(prompt)
        message_placeholder.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
