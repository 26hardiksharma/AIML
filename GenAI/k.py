from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

llm = ChatOllama(model="llama3.2", temperature=0, num_predict=512)

prompt1 = PromptTemplate(
    template="You are in a terminal environment, so generate plain text without any text decorations. Generate short and simple notes on the following topic: {topic}",
    input_variables=["topic"],
)

prompt2 = PromptTemplate(
    template="You are in a terminal environment, so generate plain text without any text decorations. Generate 5 short, clear questions about the following topic: {topic}",
    input_variables=["topic"],
)

prompt3 = PromptTemplate(
    template=(
        "Merge the following notes and questions into a concise study guide:\n\n"
        "Notes:\n{notes}\n\nQuestions:\n{questions}"
    ),
    input_variables=["notes", "questions"],
)

parser = StrOutputParser()

parallel_chain = RunnableParallel(
    {
        "notes": prompt1 | llm | parser,
        "questions": prompt2 | llm | parser,
    }
)

chain = parallel_chain | prompt3 | llm | parser

result = chain.invoke({"topic": "Photosynthesis"})
print(result)

# chain.get_graph().print_ascii()

