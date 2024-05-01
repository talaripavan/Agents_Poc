from llama_index.core import SimpleDirectoryReader ,Document

documents = SimpleDirectoryReader("data").load_data()
print("Document ID:", documents[0].doc_id)

