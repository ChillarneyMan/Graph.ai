import os
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS

def document_agent(llm, API_KEY):
    st.subheader("Upload a Document and Ask Questions")
    
    uploaded_file = st.file_uploader("Upload a text document", type=["txt"])
    
    if uploaded_file:
        text = uploaded_file.read().decode("utf-8")
        
        with open("uploaded_document.txt", "w") as f:
            f.write(text)
        
        loader = TextLoader("uploaded_document.txt")
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        documents = text_splitter.split_documents(docs)

        embeddings = OpenAIEmbeddings(openai_api_key=API_KEY)
        
        vectorstore = FAISS.from_documents(documents, embeddings)
        vectorstore.save_local("vectorstore")
        new_db = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
        
        prompt_template = """You are a storyteller and have a great way with words. {context}
        Question: {question}
        Response: """
        
        title_template = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        memory = ConversationBufferMemory(input_key='context', memory_key='chat_history')
        
        title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='response', memory=memory)
        
        prompt = st.text_input("Enter prompt here")
        
        if prompt:
            docs_with_scores = new_db.similarity_search_with_score(prompt, k=5)
            relevant_docs = [doc[0].page_content for doc in docs_with_scores]
            context = "\n\n".join(relevant_docs)
            response = title_chain.run({"context": context, "question": prompt})
            st.text_area("Response Generator", value=response, max_chars=None, height=800)
    else:
        st.write("Please upload a document:")