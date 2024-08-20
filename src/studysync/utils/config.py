import os
from typing import List, Union
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
)
from studysync.processor.conversation.prompts import (
    qna_prompt,
    cqna_prompt,
    topic_prompt,
    compare_answer_prompt,
    query_indexed_file_prompt,
)
from studysync.processor.conversation.parser import (
    qna_parser,
    cqna_parser,
    topic_parser,
    compare_answer_parser,
    string_output_parser
)

logger = logging.getLogger(__name__)


class VectorDatabase:
    def __init__(self):
        self.qdrant_client = QdrantClient(
            os.getenv("QDRANT_CLIENT_URL"), api_key=os.getenv("QDRANT_API_KEY")
        )
        self.collection_name = "studysync"
        # self.create_collection(self.collection_name)

    def retrieve_content(self, embedding_response, collection_name, docId: str = None):
        query_vector = embedding_response["embedding"]
        search_result = self.qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="document_id", match=qdrant_models.MatchValue(value=docId)
                    )
                ]
            ),
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

    def scroll_document(self, docId: str):
        search_result, _ = self.qdrant_client.scroll(
            collection_name=self.collection_name,
            with_vectors=False,
            scroll_filter=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="document_id", match=qdrant_models.MatchValue(value=docId)
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
            separator="", chunk_size=1000, chunk_overlap=200, length_function=len
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
        self.MAX_SIZE_A_PROMPT = 3000

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

    async def generate_from_doc(
        self,
        docIdList: List[str],
        maxCount: str,
        result: Union[
            CQuestionAnswerCollection
            | QuestionAnswerCollection
            | List[TopicOfStudyCollection]
        ],
        parser,
        prompt,
    ):

        for docId in docIdList:
            search_result = self.vector_database.scroll_document(docId)
            start = 0
            while start < len(search_result):
                model = await self.gemini.chat_gemini_langchain()
                contents = ""
                end = min(len(search_result), start + self.MAX_SIZE_A_PROMPT)
                for index in range(start, end):
                    contents += search_result[index].payload.get("text") + "\n"

                promt_and_model = prompt | model
                output = promt_and_model.invoke(
                    {"document_content": contents, "max_count": maxCount}
                )
                collection = parser.invoke(output)

                result.collection.append(collection.collection)
                start += self.MAX_SIZE_A_PROMPT

    async def qna_from_doc(
        self, docIdList: List[str], maxCount: str
    ) -> QuestionAnswerCollection:
        qna_list = QuestionAnswerCollection(collection=[])
        await self.generate_from_doc(
            docIdList, maxCount, qna_list, qna_parser, qna_prompt
        )

        return qna_list

    async def cqna_from_doc(
        self, docIdList: List[str], maxCount: str
    ) -> CQuestionAnswerCollection:
        cqna_list = CQuestionAnswerCollection(collection=[])
        await self.generate_from_doc(
            docIdList, maxCount, cqna_list, cqna_parser, cqna_prompt
        )

        return cqna_list

    async def topics_from_doc(
        self, docIdList: List[str], maxCount: str
    ) -> List[TopicOfStudyCollection]:
        topic_list = TopicOfStudyCollection(collection=[], collectionName="")
        await self.generate_from_doc(
            docIdList, maxCount, topic_list, topic_parser, topic_prompt
        )

        return topic_list

    async def compare_answer(self, rightAnswer: str, givenAnswer: str):
        model = await self.gemini.chat_gemini_langchain()
        promt_and_model = compare_answer_prompt | model
        output = promt_and_model.invoke(
            {"right_answer": rightAnswer, "given_answer": givenAnswer}
        )
        answer_correctness = compare_answer_parser.invoke(output)
        return answer_correctness

    async def query_indexed_file(self, query: str, fileId: str):
        embedding = await self.gemini.embed_content(
            content=query,
            task_type="retrieval_query",
        )
        retrieved_contents = self.vector_database.retrieve_content(
            embedding,
            collection_name=self.vector_database.collection_name,
            docId=fileId,
        )
        model = await self.gemini.chat_gemini_langchain()
        chain = query_indexed_file_prompt | model | string_output_parser
        response = chain.invoke(
            {"instruction": query, "context": retrieved_contents}
        )
        return response
