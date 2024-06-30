from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import CharacterTextSplitter

loader = UnstructuredFileLoader("StudySync.pdf", mode="single")
pages =  loader.load()

text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
chunks = text_splitter.split_documents(pages)

print(chunks[0].page_content)