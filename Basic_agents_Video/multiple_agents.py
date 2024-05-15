from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core import SummaryIndex
from llama_index.core.tools import QueryEngineTool
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding


from dotenv import load_dotenv
load_dotenv()


path = ["MetaGPT.pdf","paul-graham-ideas.pdf"]
documents = SimpleDirectoryReader(input_files=path).load_data()
splitter = SentenceSplitter(chunk_size=1024)
nodes = splitter.get_nodes_from_documents(documents)
#llm = OpenAI(model="gpt-3.5-turbo")
llm = OpenAI(model="gpt-4o")

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
    description=(" Useful for summarization questions related to your document."),
)
vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_query_engine,
    description=(" Useful for retrieving specific context from the document."),
)


from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner

agent_worker = FunctionCallingAgentWorker.from_tools(
    tools=[vector_tool,summary_tool],
    llm=llm,
    verbose=True
    )
agent = AgentRunner(agent_worker)
response = agent.query(
    "Tell me about the agent roles in MetaGPT, "
    "and then how they communicate with each other."
    " Summarize the Paul's Journey. "
)
""" Limitations """
# Token Limit , we can not exceed the limit upto 60000 .
print(str(response))

