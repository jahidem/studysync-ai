from typing import List
from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
from studysync.utils.models import (
    QuestionAnswerCollection,
    CQuestionAnswerCollection,
    TopicOfStudyCollection,
    AnswerCorrectness,
)

qna_parser = PydanticOutputParser(pydantic_object=QuestionAnswerCollection)
cqna_parser = PydanticOutputParser(pydantic_object=CQuestionAnswerCollection)
topic_parser = PydanticOutputParser(pydantic_object=TopicOfStudyCollection)
compare_answer_parser = PydanticOutputParser(pydantic_object=AnswerCorrectness)
string_output_parser = StrOutputParser()
