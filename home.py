import streamlit as st
from llama_index.core import VectorStoreIndex ,StorageContext
from llama_index.vector_stores.milvus import MilvusVectorStore
from PyPDF2 import PdfReader
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document
from typing import List

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# TODO :- Try to retun the Document List .
def get_chunks(docs) -> List[Document]:
    
    # If I am not declaring it as a list we would not able to chunk it in a meaning full way [18000 nodes would generate].
    text = [docs] 
    
    # If I am not converting it into documents , then we are unable to chunk it.
    documents = [Document(text=t) for t in text] 
    
    # Chunk size is 1024 --> 5 node ; 524 --> 10 nodes
    node_parser = node_parser = SentenceSplitter(chunk_size=524, chunk_overlap=20)
    nodes = node_parser.get_nodes_from_documents(documents=documents,node_parser = node_parser)
    return nodes
     
def get_milvus(chunks):
    # documents[0].get_doc_id ==> This is used to get the doc id .
    vector_store = MilvusVectorStore(dim=1536, collection_name="llamacollection" , overwrite=False)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(documents=chunks,storage_context=storage_context)
    return index

with st.sidebar:
    st.subheader("Your documents")
    pdf_docs = st.file_uploader(
        "Upload your PDF's here and click on 'Process' " , accept_multiple_files=True
    )
    if st.button("Processing"):
        docs = get_pdf_text(pdf_docs)
        chunks = get_chunks(docs)
        st.write(chunks)
        vector_store = get_milvus(chunks)
        #st.write("Stored the data into Milvus.")