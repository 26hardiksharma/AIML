from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate, load_prompt
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")

template = load_prompt('template.json')

paper_input = "Animal Farm"
style_input = "Detailed"
length = "2 Paragraphs"





chain = template | model
result = chain.invoke({
    "paper_input":paper_input,
    "style_input":style_input,
    "length_input":length
})

print(result)