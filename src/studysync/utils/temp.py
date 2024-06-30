import dotenv
import google.generativeai as gemini_client
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
import os

qdrant_client = QdrantClient(
    url="https://f758c17d-66b4-4233-8070-b5fb3f6a8f98.europe-west3-0.gcp.cloud.qdrant.io:6333", 
    api_key=os.getenv('QDRANT_API_KEY'),
)
gemini_client.configure(api_key=os.getenv('GEMINI_API_KEY'))
collection_name = "about_qdrant"


texts = [
    "Qdrant is a vector database that is compatible with Gemini.",
    "Gemini is a new family of Google PaLM models, released in December 2023.",
]

results = [
    gemini_client.embed_content(
        model="models/embedding-001",
        content=sentence,
        task_type="retrieval_document",
        title="Qdrant x Gemini",
    )
    for sentence in texts
]


points = [
    PointStruct(
        id=idx,
        vector=response['embedding'],
        payload={"text": text},
    )
    for idx, (response, text) in enumerate(zip(results, texts))
]

qdrant_client.create_collection(collection_name, vectors_config=
    VectorParams(
        size=768,
        distance=Distance.COSINE,
    )
)

qdrant_client.upsert(collection_name, points)

ou = qdrant_client.search(
    collection_name=collection_name,
    query_vector=gemini_client.embed_content(
        model="models/embedding-001",
        content="Is Qdrant compatible with Gemini?",
        task_type="retrieval_query",
    )["embedding"],
)
print(ou)


