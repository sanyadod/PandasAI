import os 
import streamlit as st

from langchain_experimental.agents import create_csv_agent
from langchain.llms import OpenAI
from dotenv import load_dotenv



def main():
    
    load_dotenv()

    # load openAI key from env
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY is not set")
        exit(1)
    else:
        print("OPENAI_API_KEY is set")
 
    #  gui setup
    st.set_page_config(page_title="Dynamic Data Insights using LLM")
    st.header("Dynamic Data Insights using LLM")
    csv_file = st.file_uploader("Upload a CSV file", type="csv")

    # csv agent
    if csv_file is not None:
        # use agent to answer the user questions
        user_question = st.text_input("Queries related to the dataset: ")

        llm = OpenAI(temperature=0) # using openAI as llm - set temp as 0 no creative answer all the same
        agent = create_csv_agent(
            llm, csv_file, verbose=True) # verbose shows the thoughts of the llm

        
        

        if user_question is not None and user_question != "":
            response = agent.run(user_question) # run instructions
            st.write(response)


if __name__ == "__main__":
    main()
