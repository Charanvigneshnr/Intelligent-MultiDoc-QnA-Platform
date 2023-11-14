import streamlit as st
from streamlit_chat import message
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
import os
import requests
from dotenv import load_dotenv
import tempfile

load_dotenv()

def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! I am ready to answer your questions about your documents."]
    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey!"]

def conversation_chat(query, chain, history):
    result = chain({"question": query, "chat_history": history})
    history.append((query, result["answer"]))

    # Print intermediate responses on the fly
    st.session_state['past'].append(query)
    st.session_state['generated'].append(result["answer"])

    return result["answer"]

def display_chat_history(chain):
    reply_container = st.container()
    container = st.container()
    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask about your Documents", key='input')
            submit_button = st.form_submit_button(label='Send')
        if submit_button and user_input:
            with st.spinner('Generating response...'):
                output = conversation_chat(user_input, chain, st.session_state['history'])
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)
    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="fun-emoji", seed="Ginger")
                message(st.session_state["generated"][i], key=str(i), avatar_style="bottts-neutral", seed="Aneka")

def create_conversational_chain(api_endpoint):
    return CTransformers(api_endpoint, max_tokens=1000)

def main():
    load_dotenv()
    initialize_session_state()
    st.title("Intelligent MultiDoc QnA Platform :books:")
    st.sidebar.title("Upload your documents to process")
    uploaded_files = st.sidebar.file_uploader("Upload files", accept_multiple_files=True)
    st.sidebar.write("After your prompt, please wait, if the response is small then it will be fast, else it will "
                     "take some time as it is generating the full string then it prints out the response")
    if uploaded_files:
        text = []
        for file in uploaded_files:
            file_extension = os.path.splitext(file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name
            loader = None
            if file_extension == ".pdf":
                loader = PyPDFLoader(temp_file_path)
            elif file_extension == ".docx" or file_extension == ".doc":
                loader = Docx2txtLoader(temp_file_path)
            elif file_extension == ".txt":
                loader = TextLoader(temp_file_path)
            if loader:
                text.extend(loader.load())
                os.remove(temp_file_path)
        text_chunks = text
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                           model_kwargs={'device': 'cpu'})

        vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)

        llama_api_endpoint = os.getenv("GCP_VERTEX_AI_ENDPOINT", "default_value_if_not_set")
        chain = create_conversational_chain(llama_api_endpoint)
        display_chat_history(chain)

if __name__ == "__main__":
    main()
