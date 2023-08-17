import os
import streamlit as st
from tempfile import NamedTemporaryFile
from streamlit_chat import message
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv 
from langchain.agents import create_csv_agent
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
# from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub


# HELPER FUNCTIONS
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=50,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base")

    # try:
    #     vectorstore = FAISS.load_local("faiss_index", embeddings)
    # except:
    #     vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    #     vectorstore.save_local("faiss_index")
    
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)

    return vectorstore

def query_agent(agent, query, chat_history):
    """
    Query an agent and return the response as a string.

    Args:
        agent: The agent to query.
        query: The query to ask the agent.
        chat_history: history of the chat for csv agent so far

    Returns:
        The response from the agent as a string.
    """

    prompt = (
        """
            If you do not know the answer, reply as "I do not know."

            if any question requires a context from previous chat history, go through the chat_history to understand the context and
            then answer the question.

            All the answers must be in string format 

            Lets think step by step.

            Below is the chat history.
            Chat_history: 
            """
        + chat_history 
        + """Below is the query.
            Query:"""
        + query
    )

    # Run the prompt through the agent.
    response = agent.run(prompt)

    return response

def generateContextHistory(histArr):
    context_chat = ""
    n = len(histArr)
    for i in range(max(n-3, 0), n):
        context_chat += histArr[i][0] + "\n"
        context_chat += histArr[i][1] + "\n"
    
    print(context_chat)
    return context_chat

def main():
    # Load environment variables from .env file
    load_dotenv()

    st.set_page_config(page_title= "ChatDoc Application 📃")

    st.header("ChatDoc Application 📃")

    user_api_key = st.sidebar.text_input(
        label="#### Your OpenAI API key 👇",
        placeholder="Paste your openAI API key, sk-",
        type="password")

    if user_api_key is not None:
        os.putenv("OPENAI_API_KEY", user_api_key)
        load_dotenv()

    st.write(user_api_key)


    with st.sidebar:
        st.subheader("Your Documents")
        uploaded_file = st.sidebar.file_uploader("Upload your document here and Click on Process Docs", accept_multiple_files= True)
        # process = st.button("Process Docs")

    if len(uploaded_file) > 0:
        doc_type_set = set()
        for file_item in uploaded_file:
            doc_type_set.add(file_item.type)
        
        if len(doc_type_set) > 1 and 'text/csv' in doc_type_set:
            st.warning(body = "Adding csv file with any other text type document is not allowed", icon = "⚠️")

        elif len(doc_type_set) == 1 and 'text/csv' in doc_type_set:
            if len(uploaded_file) > 1:
                st.warning(body = "Adding multiple csv file type document is not allowed", icon = "⚠️")
            else:
                with st.spinner("Processing"):
                    with NamedTemporaryFile(suffix ='.csv', delete=False) as f: # Create temporary file
                        f.write(uploaded_file[0].getvalue())
                        f.flush()
                    
                    # user_question = st.text_input("Ask a question about your document: ")

                    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
                    agent = create_csv_agent(llm, f.name, verbose=True)

                if 'history' not in st.session_state:
                    st.session_state['history'] = []

                if 'generated' not in st.session_state:
                    st.session_state['generated'] = ["Hello ! Ask me anything about the document 🤗"]

                if 'past' not in st.session_state:
                    st.session_state['past'] = ["Hey ! 👋"]

                #container for the chat history
                response_container = st.container()
                #container for the user's text input
                container = st.container()

                with container:
                    with st.form(key='my_form', clear_on_submit=True):
                        
                        user_input = st.text_input("Query:", placeholder="Ask a question about your document: ", key='input').strip()
                        submit_button = st.form_submit_button(label='Send')
                        
                    if submit_button and user_input:
                    
                        context_chat = generateContextHistory(st.session_state['history'])

                        output = query_agent(agent=agent, query=user_input, chat_history=context_chat)
                        
                        st.session_state['past'].append(user_input)
                        st.session_state['history'].append((user_input, output))
                        st.session_state['generated'].append(output)
                        

                if st.session_state['generated']:
                    with response_container:
                        for i in range(len(st.session_state['generated'])):
                            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                            message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")

        elif doc_type_set:
   
            # get pdf text
            raw_text = get_pdf_text(uploaded_file)
            # st.write(raw_text)

            # get the text chunks
            text_chunks = get_text_chunks(raw_text)
            # st.write(text_chunks)

            # create vector store
            vectorstore = get_vectorstore(text_chunks)
            # st.write(vectorstore)
    
            #Intialize llm-chain
            llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
            # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

            chain = ConversationalRetrievalChain.from_llm( llm=llm, retriever=vectorstore.as_retriever())
            
            if 'history' not in st.session_state:
                st.session_state['history'] = []

            if 'generated' not in st.session_state:
                st.session_state['generated'] = ["Hello ! Ask me anything about the document 🤗"]

            if 'past' not in st.session_state:
                st.session_state['past'] = ["Hey ! 👋"]

            #container for the chat history
            response_container = st.container()
            #container for the user's text input
            container = st.container()

            with container:
                with st.form(key='my_form', clear_on_submit=True):
                    
                    user_input = st.text_input("Query:", placeholder="Ask a question about your document: ", key='input')
                    submit_button = st.form_submit_button(label='Send')
                    
                if submit_button and user_input:

                    n = len(st.session_state["history"])
                    prev_chats = []
                    for i in range(max(n-5, 0), n):
                        prev_chats.append(st.session_state["history"][i])

                    result = chain({"question": user_input, "chat_history": prev_chats})


                    response = result["answer"]
                    
                    st.session_state['past'].append(user_input)
                    st.session_state['history'].append((user_input, response))
                    st.session_state['generated'].append(response)

            if st.session_state['generated']:
                with response_container:
                    print("Entered Response conatiner")
                    for i in range(len(st.session_state['generated'])):
                        message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                        message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")


if __name__ == "__main__":
    main()