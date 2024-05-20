import streamlit as st
from llama_index.core.tools import FunctionTool
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex,StorageContext
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core import SimpleDirectoryReader
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner
from llama_index.core.tools.query_engine import QueryEngineTool
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

def load_data(file_path):
    reader = SimpleDirectoryReader(input_files=[file_path])
    docs = reader.load_data()
    spilter = SentenceSplitter(chunk_size=1024)
    chunk = spilter.get_nodes_from_documents(documents=docs)
    vector_store = MilvusVectorStore(collection_name="llamacollection",dim=1536,overwrite=True,uri="")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    vector_index = VectorStoreIndex(nodes=chunk,storage_context=storage_context)
    return vector_index

def function_call(user_msg):
    llm = OpenAI(model="gpt-3.5-turbo")
    data_tool = FunctionTool.from_defaults(fn=load_data)
    success_message = llm.predict_and_call(
        tools=[data_tool],
        user_msg=user_msg,
        verbose=True
        )
    return success_message 

selected_page = st.sidebar.selectbox("Select a Page", ["Upload Document", "Chat Interface"])
if selected_page == "Upload Document":
    uploaded_file = st.file_uploader(" Upload a document ",type=["pdf", "txt", "docx","csv","xlsx"])
    if uploaded_file is not None:
        file_path = uploaded_file.name
    #st.write(file_path)
    
        with open(file_path,"wb") as f:
            f.write(uploaded_file.getvalue())
    
        vector_index = load_data(file_path)
        success_message = function_call(f"Load the data {file_path}")
        st.session_state["uploaded_filename"] = uploaded_file.name 
        st.success(f"Document uploaded {file_path} and processed successfully!")

elif selected_page == "Chat Interface":
    uploaded_filename = st.session_state.get("uploaded_filename")
    if uploaded_filename is None:
        st.error("Please upload a document first.")
    
    vector_index = load_data(uploaded_filename)  
    vector_query_engine = vector_index.as_query_engine()
    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_query_engine,
        description="Search documents based on your question."
        )
    system_prompt = """
        You are an agent designed to answer queries over a set of documents.
        Please use the provided tools to answer questions and summarize relevant information.
        Do not rely on prior knowledge.
    """
    llm = OpenAI("gpt-3.5-turbo")
    agent_worker = FunctionCallingAgentWorker.from_tools(
        tools=[vector_tool],  
        llm=llm, 
        system_prompt=system_prompt,
        verbose=True
        )
    agent = AgentRunner(agent_worker)
    
        
    user_query = st.text_input("Ask your question:", key="user_query")
    
    if user_query:
        response = agent.chat(f"User asks: {user_query}")
        st.write(response.response)
