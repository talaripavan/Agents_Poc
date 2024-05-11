import streamlit as st
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from dotenv import load_dotenv
from llama_index.core import SummaryIndex,VectorStoreIndex
from llama_index.core.tools import QueryEngineTool
load_dotenv()

def load_data(filepath):
  with st.spinner(text="Loading the document"):
    reader = SimpleDirectoryReader(input_files=[filepath])
    docs = reader.load_data()
  return docs

def chunks(documents):
    splitter = SentenceSplitter(chunk_size=1024)
    nodes = splitter.get_nodes_from_documents(documents)
    Settings.llm = OpenAI(model="gpt-3.5-turbo")
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
    return nodes
    
def summary_tool(chunk):
    summary_index = SummaryIndex(chunk)
    summary_query_engine = summary_index.as_query_engine(
    response_mode = "tree_summarize",
    use_async = True,
    )
    summary_tool = QueryEngineTool.from_defaults(
    query_engine=summary_query_engine,
    description=(" Useful for summarization questions related to your document."),
    )
    return summary_tool


uploaded_file = st.file_uploader(" Upload a document ",type=["pdf", "txt", "docx"])
if uploaded_file is not None:
    file_path = uploaded_file.name
    with open(file_path,"wb") as f:
        f.write(uploaded_file.getvalue())
    documents = load_data(file_path)
    chunk = chunks(documents)
    tool = summary_tool(chunk)