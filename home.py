from core.agent_builder.base import RAGAgentBuilder

base = RAGAgentBuilder()
load = base.load_data(file_names=["requirements.txt"])

for doc in load:
    print(doc)
