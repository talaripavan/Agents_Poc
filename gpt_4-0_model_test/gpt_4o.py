from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core import SimpleDirectoryReader

llm = OpenAIMultiModal(
    model="gpt-4o"
)

image_documents = SimpleDirectoryReader(input_files=["bow1.jpg"]).load_data()
response = llm.complete(
    prompt = "Are there 2 females in the picture ?",
    image_documents = image_documents
)

print(response)