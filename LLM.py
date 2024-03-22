import streamlit as st 
import pandas as pd
import os 

from dotenv import load_dotenv

from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
import matplotlib

matplotlib.use('TkAgg') # for plotting the graphs

# getting the openAI key
load_dotenv()
APIKEY = os.getenv("OPENAI_API_KEY")
if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY is not set")
        exit(1)
else:
    print("OPENAI_API_KEY is set")

# setup the LLM
llm = OpenAI(api_token=APIKEY) # create an LLM by instantiating OpenAI object, and passing API token
pandas_ai = PandasAI(llm) # create PandasAI object, passing the LLM

#  gui setup 
st.set_page_config(page_title="Dynamic Data Insights using LLM")
st.header("Dynamic Data Insights using LLM")
csv_file = st.file_uploader("Upload a CSV file", type="csv") #upload the csv file

if csv_file is not None:
    df = pd.read_csv(csv_file) # creating dataframe using the csv file
    prompt = st.text_area("Enter your prompt:") # gui for getting user prompt

    if st.button("Generate"): # generate output
        # check if prompt is entered
        if prompt: 
            st.write("Generating response...")
            st.write(pandas_ai.run(df,prompt=prompt)) # gets the prompts answer from the llm
        else:
            st.warning("Please enter a prompt.")
    