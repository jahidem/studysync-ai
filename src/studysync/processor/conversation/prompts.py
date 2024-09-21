from langchain.prompts import PromptTemplate
from studysync.processor.conversation.parser import (
    qna_parser,
    cqna_parser,
    topic_parser,
    compare_answer_parser,
)

## Personality
## --
personality = "You are StudySync, a large language model, be comprehensive in your responses."


qna_template = (
    personality
    + """
Generate at most {max_count} Multiple choice Question and Answers:
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
Generate at most {max_count} Question and Answers:
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
A Learner wants to know the list of concepts to study from a document.
List at most {max_count} main concepts from the Study materials:
You must fulfill these:
	1. concepts must be relevent to the given contents and important.
	2. Only include the broad concepts.

You must follow these reasoning steps:
	1. What can be the concepts for a student to better learn the contents.
	2. Now what is the concise title.
	3. What can be the short and understandable descriptions for each of the concepts.

Output format:
{format_instructions}

Study materials:

{document_content}

"""
)

compare_answer_template = (
    personality
    + """
Given the 'book content' and 'student answer' give a correctness in percentage:
You must follow these reasoning steps:
	1. Understand both the 'student answer' and 'book content'.
	2. Compare the correctness of 'student answer' with respect to the 'book content'.
  3. Treat the content of the 'book content' as fact and reason with respect of it.

book content:
{right_answer}

student answer:
{given_answer}

Output format:
{format_instructions}

"""
)


query_indexed_file = (
    personality
    + """
    1. Use the following pieces of retrieved context to answer the question.
    2. If the context is related include it in the answer otherwise don't mention anything about the context and answer from your knowledge.
Question: {instruction} 

Context: {context} 

Answer:
"""
)

content_from_topic_template = (
    personality
    + """
    
instruction:
Given study topics generate some study contents utilizing referrence contexts. Follow these steps:
  1. Make sure all the provided topics are considered.
  2. for each provided topic make a formal and comprehensive contents so that the topic can be fully understood.
  3. provide the response in proper markdown formatings.

below are the list of provided topics:
{topics}

below are the reffrence contexts to be considered:
'''
{refference}
'''
"""
)

qna_prompt = PromptTemplate(
    template=qna_template,
    input_variables=["document_content", "max_count"],
    partial_variables={"format_instructions": qna_parser.get_format_instructions()},
)

cqna_prompt = PromptTemplate(
    template=cqna_template,
    input_variables=["document_content", "max_count"],
    partial_variables={"format_instructions": cqna_parser.get_format_instructions()},
)

topic_prompt = PromptTemplate(
    template=topic_template,
    input_variables=["document_content", "max_count"],
    partial_variables={"format_instructions": topic_parser.get_format_instructions()},
)

compare_answer_prompt = PromptTemplate(
    template=compare_answer_template,
    input_variables=["right_answer", "given_answer"],
    partial_variables={
        "format_instructions": compare_answer_parser.get_format_instructions()
    },
)

query_indexed_file_prompt = PromptTemplate(
    template=query_indexed_file,
    input_variables=["instruction", "context"],
    partial_variables={
        "format_instructions": compare_answer_parser.get_format_instructions()
    },
)

content_from_topic_prompt = PromptTemplate(
  template=content_from_topic_template,
    input_variables=["topics", "refference"],
)