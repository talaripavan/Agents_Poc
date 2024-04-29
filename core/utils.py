from llama_index.core import SimpleDirectoryReader , Document
from typing import Optional , List
from dotenv import load_dotenv

load_dotenv()

# To load the data we need to specify the data folder in the Core.

def load_data(
    file_names : Optional[List[str]] = None,
    directory : Optional[str] = None,
) -> List[Document]:
    file_names = file_names or []
    directory = directory or ""
    
    num_specified = sum(1 for v in [file_names,directory] if v)
    
    if num_specified == 0:
        raise ValueError(" Must specify either file_names or directory")
    elif num_specified > 1:
        raise ValueError(" Must specify only one of the file_names or directory")
    
    elif file_names:
        reader = SimpleDirectoryReader(input_files=file_names)
        docs = reader.load_data()
    elif directory:
        reader = SimpleDirectoryReader(input_dir=directory)
        docs = reader.load_data()
    # I am not using any urls
    else:
        raise ValueError("Must specify either file_names or directory")

    return docs

'''
for doc in docs:
    print(doc) 
'''



