from langchain.prompts import PromptTemplate
from studysync.processor.conversation.parser import (
    qna_parser,
    cqna_parser,
    topic_parser,
    compare_answer_parser,
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

compare_answer_template = (
    personality
    + """
Given the 'fixed answers' and 'student given answer' given a correctness in percentage:
You must follow these reasoning steps:
	1. Understand both the 'student given answer' and 'fixed answers'.
	2. Compare the correctness of 'student given answer' with respect to the 'fixed answers'.
  3. Treat the content of the 'fixed answers' as fact and reason with respect of it.

'fixed answers':
{right_answer}

'student given answer':
{given_answer}

Output format:
{format_instructions}

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

compare_answer_prompt = PromptTemplate(
    template=compare_answer_template,
    input_variables=["right_answer", "given_answer"],
    partial_variables={
        "format_instructions": compare_answer_parser.get_format_instructions()
    },
)
