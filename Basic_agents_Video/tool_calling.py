""" In the Tool calling function, Here we declare a function to call then choose the correct function tool according to the user query . 

Steps :-
1. Define a function.
2. Declare the FunctionTool.
3. Declare the LLM and use [ predict_and_call ] method .

"""

from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from typing import List
'''
""" The Basic Understanding of Function tool Calling. """

def add(x:int , y:int) -> int:
    """ Adds two intergers together. """
    return x + y

def mystery(x:int, y:int) -> int:
    """ Mystery function that operated on top of two numbers. """ 
    return (x+y) * (x+y)

def multipy(x:int, y:int) -> int:
    """ Multipy the two numbers. """
    return x * y
    
add_tool = FunctionTool.from_defaults(fn=add)
mystery_tool = FunctionTool.from_defaults(fn=mystery)
multipy_tool = FunctionTool.from_defaults(fn=multipy)

llm = OpenAI(model="gpt-3.5-turbo")
response = llm.predict_and_call(
    [add_tool,mystery_tool,multipy_tool],
    user_msg = "Create an agent with name 'Pavan' with the chunk_size 524 ",
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
from llama_index.core.vector_stores import FilterCondition
from llama_index.core import SummaryIndex
from llama_index.core.tools import QueryEngineTool


documents = SimpleDirectoryReader(input_files=["paul-graham-ideas.pdf"]).load_data()
spillter = SentenceSplitter(chunk_size=1024)
nodes = spillter.get_nodes_from_documents(documents)
#print(nodes[0].get_content(metadata_mode="all")) # Question :- What is the most question asked by the people to the author ?
vector_index = VectorStoreIndex(nodes)

""" Here we are declaring the Metadata Filters by declaring page_number but we need to create a function which is responsible for declaring the page_numbers by their own according to the user query. """

'''
query_engine = vector_index.as_query_engine(
    similarity_top_k = 2,
    filters = MetadataFilters.from_dicts(
        [
            {"key": "page_label", "value" : "1"}
        ]
    )
)
response = query_engine.query("What is the most question asked by the people to the author and what did he answered ? ")
print(str(response))
for n in response.source_nodes:
    print(n.metadata)
'''

# Function which is responsible for Metadata Filters .
def vector_query(query:str,page_numbers:List[str]) -> str:
    metadata_dicts = [
        {"key": "page_label", "value":p} for p in page_numbers
    ]
    query_engine = vector_index.as_query_engine(
        similarity_top_k = 2,
        filters = MetadataFilters.from_dicts(
            metadata_dicts,
            condition=FilterCondition.OR
        )
    )
    response = query_engine.query(query)
    return response

vector_query_tool = FunctionTool.from_defaults(
    name="vector_tool",
    fn = vector_query
)

summary_index = SummaryIndex(nodes)
summary_query_engine = summary_index.as_query_engine(
    response_mode = "tree_summarize",
    use_async = True,
)
summary_tool = QueryEngineTool.from_defaults(
    name="summary_tool",
    query_engine=summary_query_engine,
    description=(
        " Useful if you want to get a summary of the document . "
    ),
)

llm = OpenAI(model="gpt-3.5-turbo",temperature=0)
'''
response = llm.predict_and_call(
    tools=[vector_query_tool,summary_tool],
    user_msg=" What is the most question asked by the people to the author and what did he answered ? ",
    verbose=True,
    )
'''
response = llm.predict_and_call(
    tools=[vector_query_tool,summary_tool],
    #user_msg= " Summarize the document ", # ==> Shows an error why ? :- You need to give correct query , Here ! prompt comes into play .
    #user_msg= "Summarize the paul-graham-ideas.pdf",
    #user_msg = " Summarize about author's Journey " ,
    user_msg=" What is the most question asked by the people to the author and what did he answered ? ",
    verbose=True,
    )