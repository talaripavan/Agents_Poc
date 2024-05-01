from core.agent_builder.base import RAGAgentBuilder
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core import StorageContext , VectorStoreIndex

load = RAGAgentBuilder()
agent = load.load_data(directory="data")
vector_store = MilvusVectorStore(dim=1536, collection_name="llamacollection" , overwrite=False)
print("Milvus Created")
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents=agent, storage_context=storage_context
)
print("Vector Index")
