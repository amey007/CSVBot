import streamlit as st 
from pandasai.llm.openai import OpenAI
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os
import pandas as pd
from pandasai import PandasAI

def main():

    load_dotenv()

    openai_api_key = os.getenv("OPENAI_API_KEY")

    def chat_with_csv(df,prompt):
        llm = OpenAI()
        pandas_ai = PandasAI(llm = llm, verbose= True)
        result = pandas_ai.run(df, prompt=prompt)
        print(result)
        return result

    st.set_page_config(page_title= "ChatDoc Application - PandasAI 📃", layout='wide')

    st.header("ChatDoc Application - PandasAI 📃")

    st.write(openai_api_key)

    # st.title("ChatCSV powered by LLM")

    input_csv = st.file_uploader("Upload your CSV file", type=['csv'])

    if input_csv is not None:

            col1, col2 = st.columns([1,1])

            with col1:
                st.info("CSV Uploaded Successfully")
                data = pd.read_csv(input_csv)
                st.dataframe(data, use_container_width=True)

            with col2:

                st.info("Chat Below")
                
                input_text = st.text_area("Enter your query")

                if input_text is not None:
                    if st.button("Chat with CSV"):
                        st.info("Your Query: "+input_text)
                        result = chat_with_csv(data, input_text)
                        st.success(result)

if __name__ == "__main__":
    main()