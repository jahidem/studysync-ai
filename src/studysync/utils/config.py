import os
from typing import List
import uuid
import aiofiles
import logging
from fastapi import File, UploadFile
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import CharacterTextSplitter
from qdrant_client import models as qdrant_models

from studysync.utils.models import (
    QuestionAnswerCollection,
    CQuestionAnswerCollection,
    TopicOfStudyCollection,
    AnswerCorrectness,
)
from studysync.processor.conversation.prompts import (
    qna_prompt,
    cqna_prompt,
    topic_prompt,
    compare_answer_prompt,
)
from studysync.processor.conversation.parser import (
    qna_parser,
    cqna_parser,
    topic_parser,
    compare_answer_parser,
)

logger = logging.getLogger(__name__)


class VectorDatabase:
    def __init__(self):
        self.qdrant_client = QdrantClient(
            os.getenv("QDRANT_CLIENT_URL"), api_key=os.getenv("QDRANT_API_KEY")
        )
        self.collection_name = "studysync"
        # self.create_collection(self.collection_name)

    def retrieve_content(self, embedding_response, collection_name, query_filter=None):
        query_vector = embedding_response["embedding"]
        search_result = self.qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
        )

        return search_result

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

    def scroll_document_list(self, docIdList: List[str]):
        search_result, _ = self.qdrant_client.scroll(
            collection_name=self.collection_name,
            with_vectors=False,
            scroll_filter=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="document_id", match=qdrant_models.MatchAny(any=docIdList)
                    )
                ]
            ),
        )
        return search_result


class IndexContent:
    def __init__(self, gemini):
        self.vectorDatabase = VectorDatabase()
        self.gemini = gemini

    def extract_text_from_file(self, file_name: str):
        loader = UnstructuredFileLoader(f"uploads/{file_name}", mode="single")
        pages = loader.load()

        text_splitter = CharacterTextSplitter(
            separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
        )
        chunks = text_splitter.split_documents(pages)
        return chunks

    async def embedding_from_text(self, content: str):
        embedding = await self.gemini.embed_content(
            content=content, task_type="retrieval_document"
        )
        return embedding

    async def run(self, file_name: str, group_uuid: str):
        chunks = self.extract_text_from_file(file_name)
        embeddings = [
            await self.embedding_from_text(document.page_content) for document in chunks
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
    def __init__(self, vector_database: VectorDatabase, gemini):
        self.vector_database = vector_database
        self.gemini = gemini

    async def get_response(self, prompt: str) -> str:
        return await self.gemini.generate_content(prompt)

    async def get_response_on_image(
        self, prompt: str, imageFile: UploadFile = File(...)
    ) -> str:
        image_bytes = await imageFile.read()
        response = await self.gemini.generate_content(
            [
                prompt,
                {
                    "mime_type": imageFile.content_type,
                    "data": image_bytes,
                },
            ]
        )
        return response.text

    def qna_from_doc(self, docIdList: List[str]) -> QuestionAnswerCollection:
        search_result = self.vector_database.scroll_document_list(docIdList)

        MAX_SIZE_A_PROMPT = 3000
        qna_list = QuestionAnswerCollection(collection=[])
        start = 0
        while start < len(search_result):
            contents = ""
            end = min(len(search_result), start + MAX_SIZE_A_PROMPT)
            for index in range(start, end):
                contents += search_result[index].payload.get("text") + "\n"
            promt_and_model = qna_prompt | self.gemini.model_langchain
            output = promt_and_model.invoke({"document_content": contents})
            qna_collection = qna_parser.invoke(output)
            qna_list.collection.append(qna_collection.collection)
            start += MAX_SIZE_A_PROMPT

        return qna_list

    def cqna_from_doc(self, docIdList: List[str]) -> CQuestionAnswerCollection:
        search_result = self.vector_database.scroll_document_list(docIdList)

        MAX_SIZE_A_PROMPT = 3000
        cqna_list = CQuestionAnswerCollection(collection=[])
        start = 0
        while start < len(search_result):
            contents = ""
            end = min(len(search_result), start + MAX_SIZE_A_PROMPT)
            for index in range(start, end):
                contents += search_result[index].payload.get("text") + "\n"
            promt_and_model = cqna_prompt | self.gemini.model_langchain
            output = promt_and_model.invoke({"document_content": contents})
            cqna_collection = cqna_parser.invoke(output)
            cqna_list.collection.append(cqna_collection.collection)
            start += MAX_SIZE_A_PROMPT

        return cqna_list

    def topics_from_doc(self, docIdList: List[str]) -> List[TopicOfStudyCollection]:
        search_result = self.vector_database.scroll_document_list(docIdList)
        MAX_SIZE_A_PROMPT = 3000
        topic_list = TopicOfStudyCollection(collection=[], collectionName="")
        start = 0
        while start < len(search_result):
            contents = ""
            end = min(len(search_result), start + MAX_SIZE_A_PROMPT)
            for index in range(start, end):
                contents += search_result[index].payload.get("text") + "\n"
            promt_and_model = topic_prompt | self.gemini.model_langchain
            output = promt_and_model.invoke({"document_content": contents})
            topic_collection = topic_parser.invoke(output)
            topic_list.collection.append(topic_collection.collection)
            topic_list.collectionName = topic_collection.collectionName
            start += MAX_SIZE_A_PROMPT

        return topic_list

    def compare_answer(self, rightAnswer: str, givenAnswer: str):
        promt_and_model = compare_answer_prompt | self.gemini.model_langchain
        output = promt_and_model.invoke(
            {"right_answer": rightAnswer, "given_answer": givenAnswer}
        )
        answer_correctness = compare_answer_parser.invoke(output)
        return answer_correctness
