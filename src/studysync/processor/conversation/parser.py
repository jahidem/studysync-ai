from typing import List
from langchain.output_parsers import PydanticOutputParser
from studysync.utils.models import (
    QuestionAnswerCollection,
    CQuestionAnswerCollection,
    TopicOfStudyCollection,
)

qna_parser = PydanticOutputParser(pydantic_object=QuestionAnswerCollection)
cqna_parser = PydanticOutputParser(pydantic_object=CQuestionAnswerCollection)
topic_parser = PydanticOutputParser(pydantic_object=TopicOfStudyCollection)
