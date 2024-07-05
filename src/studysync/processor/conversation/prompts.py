from langchain.prompts import PromptTemplate
from studysync.processor.conversation.parser import qna_parser

## Personality
## --
personality = "You are StudySync, a large language model, informative and comprehensive in your responses. You are helping a student study with some documents"


qna_template = (
    personality
    + """

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
qna_prompt = PromptTemplate(
    template=qna_template,
    input_variables=["document_content"],
    partial_variables={"format_instructions": qna_parser.get_format_instructions()},
)
