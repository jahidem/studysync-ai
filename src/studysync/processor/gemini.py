from langchain_google_genai import ChatGoogleGenerativeAI
import os


def generateResponse(prompt: str) -> str:
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", google_api_key=os.getenv("GEMINI_API_KEY")
    )
    result = llm.invoke(prompt)
    return result.content


# model = ChatGoogleGenerativeAI(
#     "gemini-1.5-flash", google_api_key=os.getenv("GEMINI_API_KEY")
# )

# import google.generativeai as genai

# model = "models/embedding-001"
# embedding = genai.embed_content(model=model, content="za wardu")
