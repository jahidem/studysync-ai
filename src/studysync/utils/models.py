from typing import List, Union
from langchain.pydantic_v1 import BaseModel, Field, validator


class QuestionAnswer(BaseModel):
    question:str = Field(description="Question the student will answer")
    choice: List[str] = Field(description="Multiple choices for the question")
    isChoiceAnswer: List[bool] = Field(description="If the corresponding index in choice is right")
   
class QuestionAnswerCollection(BaseModel):
    collection: List[QuestionAnswer] 