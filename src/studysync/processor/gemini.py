import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
import time


class RateLimit:
    """Simple Asynchronous Rate Limiter class"""

    def __init__(self, limit, period):
        self.limit = limit
        self.period = period
        self.last_used = time.time()

    async def wait_until_allowed(self):
        now = time.time()
        elapsed = now - self.last_used
        if elapsed < self.period:
            delay = self.period - elapsed
            await asyncio.sleep(delay)
        self.last_used = now


class Gemini:
    def __init__(self):
        self.embedding_model = "models/embedding-001"
        self.model = genai.GenerativeModel("gemini-1.5-flash")
        self.model_langchain = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
        self.rate_limiter = RateLimit(limit=15, period=60)  # 15 requests per minute

    async def embed_content(self, content: str, task_type="retrieval_query"):
        await self.rate_limiter.wait_until_allowed()
        return genai.embed_content(
            model=self.embedding_model, content=content, task_type=task_type
        )

    async def generate_content(self, prompt: str):
        await self.rate_limiter.wait_until_allowed()
        response = self.model.generate_content(prompt).text
        return response

    async def chat_gemini_langchain(self):
        await self.rate_limiter.wait_until_allowed()
        return self.model_langchain
