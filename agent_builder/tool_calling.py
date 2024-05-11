from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI

''' 
""" The Basic Understanding of Function tool Calling. """

def add(x:int , y:int) -> int:
    """ Adds two intergers together. """
    return x + y

def mystery(x:int, y:int) -> int:
    """ Mystery function that operated on top of two numbers. """ 
    return (x+y) * (x+y)

def multipy(x:int, y:int) -> int:
    """ Multipy the two numbers """
    return x * y

# TODO :- Create an Agent Function with parameters[name,llm,etc]

add_tool = FunctionTool.from_defaults(fn=add)
mystery_tool = FunctionTool.from_defaults(fn=mystery)
multipy_tool = FunctionTool.from_defaults(fn=multipy)

llm = OpenAI(model="gpt-3.5-turbo")
response = llm.predict_and_call(
    [add_tool,mystery_tool,multipy_tool],
    user_msg = "Tell me the output of the sum function on 3 and 9",
    #user_msg = input(), # ==> If user wants to ask a question from the terminal . 
    verbose=True
)
print(str(response))
'''

# RAG Auto-Retrieval Tool.
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.vector_stores import MetadataFilters


documents = SimpleDirectoryReader(input_files=["paul-graham-ideas.pdf"]).load_data()
spillter = SentenceSplitter(chunk_size=1024)
nodes = spillter.get_nodes_from_documents(documents)
print(nodes[0].get_content(metadata_mode="all"))