import logging
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
logger = logging.getLogger('uvicorn.error')


class Gemini:
    def __init__(self):
        self.embedding_model = "models/embedding-001"
        self.model = genai.GenerativeModel("gemini-1.5-flash")
        self.model_langchain = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

    async def embed_content(self, content: str, task_type="retrieval_query"):
        return genai.embed_content(
            model=self.embedding_model, content=content, task_type=task_type
        )

    async def generate_content(self, prompt: str):
        response = self.model.generate_content(prompt).text
        return response

    async def chat_gemini_langchain(self):
        return self.model_langchain
