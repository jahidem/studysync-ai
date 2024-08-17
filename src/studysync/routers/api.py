from ntpath import join
import os
from typing import List
from fastapi import APIRouter, File, HTTPException, UploadFile, status, Body, status
from studysync.utils.state import (
    file_handling,
    vector_database,
    index_content,
    generator,
)
from studysync.utils.state import gemini
from fastapi.responses import FileResponse
import logging

logger = logging.getLogger(__name__)
api = APIRouter()


@api.get("/generate/response")
async def generate(prompt: str) -> str:
    response = await generator.get_response(prompt)
    return response


@api.post("/generate/askImage")
async def generate(prompt: str, imageFile: UploadFile = File(...)):
    ALLOWED_IMAGE_TYPES = [
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/heic",
        "image/heif",
    ]
    if imageFile.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type. Only these image types are allowed: {', '.join(ALLOWED_IMAGE_TYPES)}",
        )

    response = await generator.get_response_on_image(prompt, imageFile)
    return response


@api.post("/uploadFile")
async def upload_file(in_file: UploadFile = File(...)):
    file_id = await file_handling.upload_file(in_file)
    return {"fileId": file_id}


@api.get("/downloadFile")
async def upload_file(uuidFileName: str):
    """
    uuidFileName=file_uuid.pdf
    """
    file_path = os.getcwd() + f"/uploads/{uuidFileName}"
    return FileResponse(path=file_path, filename=uuidFileName)


@api.delete("/removeFile")
async def delete_file(uuidFileName: str):
    """
    uuidFileName=file_uuid.pdf
    """
    file_path = join(os.getcwd(), "uploads", uuidFileName)
    os.remove(file_path)
    return {"fileId": uuidFileName}


@api.get("/indexFile")
async def upload_file(uuidFileName: str):
    """
    uuidFileName=file_uuid.pdf
    """
    await index_content.run(uuidFileName, "file")
    return {"fileId": uuidFileName}


@api.post("/queryFile")
async def query_file(query: str, in_file: UploadFile = File(...)):
    file_id = await file_handling.upload_file(in_file)
    await index_content.run(f"{file_id}.{in_file.filename.split('.')[-1]}", "emon")
    embedding = await gemini.embed_content(
        content=query,
        task_type="retrieval_query",
    )
    retrieved_contents = vector_database.retrieve_content(
        embedding,
        collection_name=vector_database.collection_name,
    )
    return {"retrieved": retrieved_contents}


@api.post("/generate/qna")
async def generate(fileId: List[str], maxCount: str = 20):
    mcqList = await generator.qna_from_doc(fileId, maxCount)
    return mcqList


@api.post("/generate/cqna")
async def generate(fileId: List[str], maxCount: str = 20):
    return await generator.cqna_from_doc(fileId, maxCount)


@api.post("/generate/compare_answer")
async def generate(rightAnswer: str = Body(None), givenAnswer: str = Body(None)):
    if rightAnswer and givenAnswer:
        return await generator.compare_answer(rightAnswer, givenAnswer)
    return {
        "message": "Both right and student answer must be given"
    }, status.HTTP_406_NOT_ACCEPTABLE


@api.post("/extract/topics")
async def generate(fileId: List[str], maxCount: str = 10):
    return await generator.topics_from_doc(fileId, maxCount)
