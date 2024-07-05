from typing import List
from langchain.output_parsers import PydanticOutputParser
from studysync.utils.models import QuestionAnswerCollection

qna_parser = PydanticOutputParser(pydantic_object=QuestionAnswerCollection)