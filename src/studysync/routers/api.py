from fastapi import APIRouter, File, UploadFile
from studysync.processor.gemini import generateResponse
from studysync.utils.state import file_handling, vector_database, index_content

api = APIRouter()


@api.post("/generate")
def generate(prompt: str):
    return generateResponse(prompt)


@api.post("/uploadFile")
async def upload_file(in_file: UploadFile = File(...)):
    file_id = file_handling.upload_file(in_file)
    return {"fileId": file_id}


@api.post("/queryFile")
async def query_file(query: str, in_file: UploadFile = File(...)):
    file_id = await file_handling.upload_file(in_file)
    index_content.run(f"{file_id}.{in_file.filename.split('.')[-1]}", "emon")
    retrieved_contents = vector_database.retrieve_content(
        query,
        collection_name=vector_database.collection_name,
    )
    return {"retrieved": retrieved_contents}
