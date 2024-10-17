import streamlit as st
from langchain_experimental.agents.agent_toolkits import create_csv_agent

def csv_agent(llm, csv_path):
    st.subheader("Ask Questions About the CSV Data")
    agent = create_csv_agent(llm, csv_path, verbose=True, allow_dangerous_code=True)
    
    user_question = st.text_input("Ask a question about your CSV: ")
    
    if user_question:
        with st.spinner(text="In progress..."):
            st.write(agent.run(user_question))