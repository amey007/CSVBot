import streamlit as st
from langchain.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
from dotenv import load_dotenv
from tempfile import NamedTemporaryFile
import pandas as pd

def main():
    # Your Code goes in here
    load_dotenv()

    st.set_page_config(page_title= "ChatDoc Application - Pandas df Agent 📃")

    st.header("ChatDoc Application - Pandas df Agent 📃")

    user_document = st.file_uploader(label="Upload your document", type="csv")

    if user_document is not None:

        with NamedTemporaryFile(suffix ='.csv', delete=False) as f: # Create temporary file
            f.write(user_document.getvalue())
            f.flush()
        
        df = pd.read_csv(f.name)
        df["Type 2"].fillna(df["Type 1"], inplace = True)
        
        user_question = st.text_input("Ask a question about your document: ")

        # llm = OpenAI(temperature=0)
        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
        agent = create_pandas_dataframe_agent(llm= llm, df = df,  verbose=True,  agent_type=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION)

        if user_question is not None and user_question.strip() != "":
            response = agent.run(user_question)
            st.write(response)


if __name__ == "__main__":
    main()