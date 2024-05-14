import streamlit as st
from llama_index.core import SimpleDirectoryReader
from dotenv import load_dotenv
load_dotenv()

def load_data(filepath):
  with st.spinner(text="Loading the document"):
    reader = SimpleDirectoryReader(input_files=[filepath])
    docs = reader.load_data()
  return docs

# It stores the files into our current directory. 
uploaded_file = st.file_uploader(" Upload a document ",type=["pdf", "txt", "docx","csv","xlsx"])
if uploaded_file is not None:
    file_path = uploaded_file.name
    st.write(file_path)
    with open(file_path,"wb") as f:
        f.write(uploaded_file.getvalue())
    documents = load_data(file_path)
    st.write(documents)