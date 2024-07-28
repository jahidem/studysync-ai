from typing import List, Union
from langchain.pydantic_v1 import BaseModel, Field, validator


class QuestionAnswer(BaseModel):
    question: str = Field(description="Question the student will answer")
    choice: List[str] = Field(description="Multiple choices for the question")
    isChoiceAnswer: List[bool] = Field(
        description="If the corresponding index in choice is right"
    )


class QuestionAnswerCollection(BaseModel):
    collection: List[QuestionAnswer]


class CQuestionAnswer(BaseModel):
    question: str = Field(description="Question the student will answer")
    answer: str = Field(description="Answer for the question")


class CQuestionAnswerCollection(BaseModel):
    collection: List[CQuestionAnswer]


class TopicOfStudy(BaseModel):
    name: str = Field(description="The concise name of the topic of study")
    desciption: str = Field(description="Short description of the topic")


class TopicOfStudyCollection(BaseModel):
    collectionName: str = Field(description="Adequate title for the topics included")
    collection: List[TopicOfStudy]


class AnswerCorrectness(BaseModel):
    correctness: int = Field(
        description="The correctness of the answer given by student in integer ranging from 0 to 100"
    )
    comment: str = Field(
        description="Concisely mention the right or wrong parts in coparison to book content."
    )
    
