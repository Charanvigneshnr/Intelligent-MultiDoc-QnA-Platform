"""
Import necessary libraries and modules for the application, including Streamlit for the user
interface, langchain for natural language processing, and other utilities like dotenv and temp file.
"""
import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Replicate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os
from dotenv import load_dotenv
import tempfile

load_dotenv()

"""
Define a function initialize_session_state to initialize session state variables if they are not present.
These variables are used to store chat history and generated responses.
"""


def initialize_session_state():
	if 'history' not in st.session_state:
		st.session_state['history'] = []
	if 'generated' not in st.session_state:
		st.session_state['generated'] = ["Hello! I am ready to answer your questions about your documents."]
	if 'past' not in st.session_state:
		st.session_state['past'] = ["Hey!"]


"""
Define a function conversation_chat to handle the conversation chat. It takes a query, a conversational chain,
and chat history as input. It updates the history with the query and its corresponding answer and updates session
state with intermediate responses.
"""


def conversation_chat(query, chain, history):
	result = chain({"question": query, "chat_history": history})
	history.append((query, result["answer"]))
	st.session_state['past'].append(query)
	st.session_state['generated'].append(result["answer"])
	return result["answer"]


"""
Define a function display_chat_history to display the chat history using Streamlit components. It includes a form
with a text input for user questions and a button to submit. The generated chat history is displayed using the
message component.
"""


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
				message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="fun-emoji",
				        seed="Ginger")
				message(st.session_state["generated"][i], key=str(i), avatar_style="bottts-neutral", seed="Aneka")


"""
Define a function create_conversational_chain to create a conversational chain. It uses the Replicate model with
specific configurations, a vector store for document retrieval, and a conversation buffer memory to store chat
history.
"""


def create_conversational_chain(vector_store):
	load_dotenv()
	llm = Replicate(
			streaming=True,
			model="replicate/llama-2-70b-chat:58d078176e02c219e11eb4da5a02a7830a283b14cf8f94537af893ccff5ee781",
			callbacks=[StreamingStdOutCallbackHandler()],
			input={"temperature": 0.01, "max_length": 500, "top_p": 1})
	memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
	chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
	                                              retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
	                                              memory=memory)
	return chain


"""
Define the main function main to run the Streamlit application. It initializes session state, sets up the UI with
file upload options, processes uploaded documents, creates a conversational chain, and displays the chat history. The
application is run when the script is executed directly.
"""


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
		text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=100, length_function=len)
		text_chunks = text_splitter.split_documents(text)
		embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
		                                   model_kwargs={'device': 'cpu'})
		
		vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)
		chain = create_conversational_chain(vector_store)
		display_chat_history(chain)


if __name__ == "__main__":
	main()
