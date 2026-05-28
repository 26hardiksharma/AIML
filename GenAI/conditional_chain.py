from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableParallel,RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

class Feedback(BaseModel):
    sentiment: Literal["Positive", "Negative", "Neutral"] = Field(
        description="Sentiment of the feedback"
    )

parser = PydanticOutputParser(pydantic_object=Feedback)

prompt = PromptTemplate(
    template="""
Classify the sentiment of the text.

{format_instructions}

Text: {text}
""",
    input_variables=["text"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)
structured_model = llm.with_structured_output(Feedback)

prompt2 = PromptTemplate(
    template= """
    Write an appropriate response to this positive feedback:\n {text}
    """,

    input_variables=["text"]
)
prompt3 = PromptTemplate(
    template= """
    Write an appropriate response to this negative feedback:\n {text}
    """,

    input_variables=["text"]
)
prompt4 = PromptTemplate(
    template= """
    Write an appropriate response to this neutral feedback:\n {text}
    """,

    input_variables=["text"]
)

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "Positive", prompt2 | llm),
    (lambda x : x.sentiment == "Negative", prompt3 | llm),
    (lambda x: x.sentiment == "Neutral", prompt4 | llm),
    RunnableLambda(lambda x: "Fallback: Couldnt Find sentiment!!!")
)


classifier = prompt | structured_model

chain = classifier | branch_chain

result = chain.invoke({"text": "This product is okay"})

print(result.content)

chain.get_graph().print_ascii()
