import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
import os
import io
from PyPDF2 import PdfReader

from langchain_google_genai import ChatGoogleGenerativeAI ## LangChain wrapper for Google's Generative AI (Gemini API), allows for multi-turn conversations where the model remembers previous interactions.
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS ## perform similarity search(in internet) on large datasets of vectors(input)
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

load_dotenv()
genai.configure(api_key = os.getenv("google_API_key"))

def get_pdf_text(pdf_docs):
    text=""
    if not isinstance(pdf_docs, list):
        pdf_docs=[pdf_docs]
    for pdf in pdf_docs: # Read all the information in all the pdf's one by one
        pdf_reader = PdfReader(io.BytesIO(pdf.read()))
        #This pdf_reader will be in format of list of pages
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    
## convert text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000) 
    chunks = text_splitter.split_text(text)
    return chunks 

## convert chunks into vector embeddings 
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store=FAISS.from_texts(text_chunks,embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template=""" 
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in the provided context just say "Answer is not available in the context", don't provide the wrong answer \n\n
    Context:\n{context}?\n
    Question:\n{question}\n
    
    Answer:
    """
    
    model=ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    ## Provide the template for the prompt
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain=load_qa_chain(model, chain_type="stuff",prompt=prompt)
    return chain # a langchain pipeline is returned

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True) # load local embeddings to new_db
    ## allow_dangerous_serialization = to allow the permissions for streamlit to access local files and process them.
    docs = new_db.similarity_search(user_question)
    chain=get_conversational_chain()
    
    response = chain(
        {"input_documents":docs, "question":user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])
    
    
def main():
    st.set_page_config("Chat with Multiple PDf")
    st.header("Chat with Multiple PDF using Gemini🤖💻")
    
    user_question = st.text_input("Ask a question from the PDF's")
    
    if user_question:
        user_input(user_question)
        
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF files and click on Submit")
        if st.button("Submit"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")
                
if __name__ == "__main__":
    main()