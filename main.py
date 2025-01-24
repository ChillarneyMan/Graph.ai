import os
import pandas as pd
import streamlit as st
from Secret import API_KEY
from langchain_openai import ChatOpenAI, OpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from graph_generator import GraphGenerator
from csv_agent import csv_agent
from document_agent import document_agent

# Set up the API key
os.environ['OPENAI_API_KEY'] = API_KEY

# Load the dataset
df = pd.read_csv('data/CleanedDataOfAllStudents.csv')
df_5_rows = df.head()
csv_string = df_5_rows.to_string(index=False)

# Initialize the language model
llm_csv = ChatOpenAI(api_key=API_KEY, temperature=0.5)

# Create the prompt template for the graphing task
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You're a data visualization expert and only use the plotly graphing library. Suppose that "
            "the data is only provided as a data/CleanedDataOfAllStudents.csv file. Here are the first 5 rows of the data set: {data}. "
            "Follow the user's indications when creating the graph."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = prompt | llm_csv

# Initialize LLM for document agent
llm_doc = OpenAI(temperature=0.9, max_tokens=3000, model='gpt-4o-mini')

def main():
    st.set_page_config(page_title="Data Analysis")
    st.title("Data Analysis 📊")
    
    tab1, tab2, tab3 = st.tabs(["Graph Generator", "CSV Agent", "Document Agent"])

    with tab1:
        graph_generator = GraphGenerator(chain, csv_string)
        graph_generator.render()

    with tab2:
        csv_agent(llm_csv, 'data/CleanedDataOfAllStudents.csv')

    with tab3:
        document_agent(llm_doc, API_KEY)

if __name__ == "__main__":
    main()