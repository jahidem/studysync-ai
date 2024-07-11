from langchain.prompts import PromptTemplate
from studysync.processor.conversation.parser import (
    qna_parser,
    cqna_parser,
    topic_parser,
)

## Personality
## --
personality = "You are StudySync, a large language model, informative and comprehensive in your responses. You are helping a student study with some documents"


qna_template = (
    personality
    + """
Generate at most 10 Multiple choice Question and Answers:
You must fulfill these:
	1. Question must be relevant to the given contents and important.
	2. Answer choices for questions must be relevant to the content and at most 5.

You must follow these reasoning steps:
	1. What can be the questions for a student to better learn the contents.
	2. Now what are the Answer choices for the question.
	3. Organize Question and Answers in the instructed format.

Output format:
{format_instructions}

Study materials:

{document_content}

"""
)

cqna_template = (
    personality
    + """
Generate at most 10 Question and Answers:
You must fulfill these:
	1. Question must be relevant to the given contents, concise and important.
	2. Answer for questions must be relevant to the content and at most 10 lines.

You must follow these reasoning steps:
	1. What can be the questions for a student to better learn the contents.
	2. Now what is the concise question.
	3. Organize Question and Answers in the instructed format.

Output format:
{format_instructions}

Study materials:

{document_content}

"""
)


topic_template = (
    personality
    + """
List main topics from the Study materials concisely:
You must fulfill these:
	1. Topics must be relevent to the given contents, concise and important.
	2. Only include most include and broad topics.

You must follow these reasoning steps:
	1. What can be the topic for a student to better learn the contents.
	2. Now what is the concise title for the topics.
	3. What can be the short and understandable descriptions for each of the topics.

Output format:
{format_instructions}

Study materials:

{document_content}

"""
)

qna_prompt = PromptTemplate(
    template=qna_template,
    input_variables=["document_content"],
    partial_variables={"format_instructions": qna_parser.get_format_instructions()},
)

cqna_prompt = PromptTemplate(
    template=cqna_template,
    input_variables=["document_content"],
    partial_variables={"format_instructions": cqna_parser.get_format_instructions()},
)

topic_prompt = PromptTemplate(
    template=topic_template,
    input_variables=["document_content"],
    partial_variables={"format_instructions": topic_parser.get_format_instructions()},
)
