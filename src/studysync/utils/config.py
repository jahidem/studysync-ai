import os
from typing import Union, List
import uuid
import aiofiles
from fastapi import File, UploadFile
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import CharacterTextSplitter
import google.generativeai as genai
from studysync.utils.models import QuestionAnswer
from qdrant_client import models as qdrant_models
from langchain_google_genai import ChatGoogleGenerativeAI

from studysync.utils.models import QuestionAnswerCollection
from studysync.processor.conversation.prompts import qna_prompt
from studysync.processor.conversation.parser import qna_parser


class VectorDatabase:
    def __init__(self):
        self.embedding_model = "models/embedding-001"
        self.qdrant_client = QdrantClient(
            os.getenv("QDRANT_CLIENT_URL"), api_key=os.getenv("QDRANT_API_KEY")
        )
        self.collection_name = "studysync"
        # self.create_collection(self.collection_name)

    def retrieve_content(self, query: str, collection_name, query_filter=None):
        query_vector = self.get_query_embeddings(query)["embedding"]
        search_result = self.qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
        )

        return search_result

    def get_query_embeddings(self, text: str):
        return genai.embed_content(
            model=self.embedding_model,
            content=text,
            task_type="retrieval_query",
        )

    def get_collection(self, collection_name: str):
        collections = self.qdrant_client.get_collections()
        if collection_name in collections:
            return self.qdrant_client.get_collection(collection_name)
        else:
            print(f"Collection '{collection_name}' not found. creating...")
            return None

    def create_collection(self, collection_name: str, vector_size=768):
        collection = self.get_collection(collection_name)
        if not collection:
            collection = self.qdrant_client.create_collection(
                collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )
        return collection


class IndexContent:
    def __init__(self):
        self.embedding_model = "models/embedding-001"
        self.vectorDatabase = VectorDatabase()

    def extract_text_from_file(self, file_name: str):
        loader = UnstructuredFileLoader(f"uploads/{file_name}", mode="single")
        pages = loader.load()

        text_splitter = CharacterTextSplitter(
            separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
        )
        chunks = text_splitter.split_documents(pages)
        return chunks

    def embedding_from_text(self, content: str):
        embedding = genai.embed_content(
            model=self.embedding_model, content=content, task_type="retrieval_document"
        )
        return embedding

    def run(self, file_name: str, group_uuid: str):
        chunks = self.extract_text_from_file(file_name)
        embeddings = [
            self.embedding_from_text(document.page_content) for document in chunks
        ]

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=response["embedding"],
                payload={
                    "text": document.page_content,
                    "group_id": group_uuid,
                    "document_id": file_name.split(".")[0],
                    "chunk_no": index,
                },
            )
            for index, (response, document) in enumerate(zip(embeddings, chunks))
        ]
        self.vectorDatabase.qdrant_client.upsert(
            self.vectorDatabase.collection_name, points
        )


class FileHandling:
    def __init__(self):
        self.uploads_dir = "uploads"
        os.makedirs(self.uploads_dir, exist_ok=True)  # create directory if not exists

    async def upload_file(self, in_file: UploadFile = File(...)):
        # Generate a secure and unique filename
        import uuid

        file_id = uuid.uuid4()
        out_file_path = f"uploads/{file_id}.{in_file.filename.split('.')[-1]}"

        # Asynchronous file processing with error handling
        try:
            async with aiofiles.open(out_file_path, "wb") as out_file:
                content = await in_file.read()
                await out_file.write(content)
        except Exception as e:
            print(f"Error saving file: {e}")  # Log the error for debugging
            return None

        return file_id


class Generator:
    def __init__(self, vector_database: VectorDatabase):
        self.model = genai.GenerativeModel("gemini-1.5-flash")
        self.model_langchain = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
        self.vector_database = vector_database

    def get_response(self, prompt: str) -> str:
        return self.model.generate_content(prompt).text

    async def get_response_on_image(
        self, prompt: str, imageFile: UploadFile = File(...)
    ) -> str:
        image_bytes = await imageFile.read()
        response = self.model.generate_content(
            [
                prompt,
                {
                    "mime_type": imageFile.content_type,
                    "data": image_bytes,
                },
            ]
        )
        return response.text

    def qna_from_doc(
        self, docIdList: List[str]
    ) -> List[QuestionAnswerCollection]:
        search_result, _ = self.vector_database.qdrant_client.scroll(
            collection_name=self.vector_database.collection_name,
            with_vectors=False,
            scroll_filter=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="document_id", match=qdrant_models.MatchAny(any=docIdList)
                    )
                ]
            ),
        )

        MAX_SIZE_A_PROMPT = 3000
        qna_list: List[QuestionAnswerCollection] = []
        start = 0
        while start < len(search_result):
            contents = ""
            end = min(len(search_result), start + MAX_SIZE_A_PROMPT)
            for index in range(start, end):
                contents += search_result[index].payload.get("text") + "\n"
            promt_and_model = qna_prompt | self.model_langchain
            output = promt_and_model.invoke({"document_content": contents})
            qna_collection = qna_parser.invoke(output)
            print(qna_collection)
            qna_list.append(qna_collection)
            start += MAX_SIZE_A_PROMPT

        return qna_list