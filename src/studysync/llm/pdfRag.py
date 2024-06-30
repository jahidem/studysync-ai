from langchain.document_loaders import PyPDFDirectoryLoader

DATA_PATH = "data"

def load_documents():
  document_loader = PyPDFDirectoryLoader(DATA_PATH)
  return document_loader.load()

documents = load_documents()
print(documents[0])