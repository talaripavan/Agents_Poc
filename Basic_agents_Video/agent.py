import streamlit as st
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core import SummaryIndex
from llama_index.core.tools import QueryEngineTool
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner

from dotenv import load_dotenv
load_dotenv()

def load_data(filepath):
  with st.spinner(text="Loading the document"):
    reader = SimpleDirectoryReader(input_files=[filepath])
    docs = reader.load_data()
    spiltter = SentenceSplitter(chunk_size=1024)
    nodes = spiltter.get_nodes_from_documents(docs)    
    llm = OpenAI(model="gpt-3.5-turbo")
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
    
    
    summary_index = SummaryIndex(nodes)
    vector_index = VectorStoreIndex(nodes)
    summary_query_engine = summary_index.as_query_engine(
        response_mode = "tree_summarize",
        use_async = True,
        )
    vector_query_engine = vector_index.as_query_engine()
    summary_tool = QueryEngineTool.from_defaults(
        query_engine=summary_query_engine,
        description=(" Use ONLY IF you want to get a holistic summary of the documents. DO NOT USE if you have specified questions over the documents."),
    )
    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_query_engine,
        description=("Utilize this tool to efficiently retrieve documents based on specified criteria."),
    )
    agent_worker = FunctionCallingAgentWorker.from_tools(
    tools=[vector_tool,summary_tool],
    llm=llm,
    system_prompt="""
    You are an agent designed to answer queries over a set of given papers.
    Please always use the tools provided to answer a question.Do not rely on prior knowledge.""",
    verbose=True
    )
    agent = AgentRunner(agent_worker)
    return agent

uploaded_file = st.file_uploader(" Upload a document ",type=["pdf", "txt", "docx","csv","xlsx"])
if uploaded_file is not None:
    file_path = uploaded_file.name
    st.write(file_path)
    with open(file_path,"wb") as f:
        f.write(uploaded_file.getvalue())
    agent = load_data(file_path)
    
    if input := st.chat_input("Your question"):
        st.session_state.messages.append({"role":"user", "content":input})

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
        {"role" : "assistant" , "content" : "Ask me a question !"}
        ]
        
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking.."):
                response = agent.chat(input)
                st.write(response.response)
