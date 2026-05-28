from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os

os.environ["USE_TF"] = "0"
os.environ["HF_HOME"] = "D:/HuggingFace_Cache"
llm = HuggingFacePipeline.from_model_id(
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task = "text-generation",
    pipeline_kwargs = dict(
        temperature = 0.5,
        max_new_tokens = 100
    )
)

chat = []
model = ChatHuggingFace(llm = llm)
while True:
    inp = input("User: ")
    if(inp.lower() == "exit"):
        break
    chat.append(inp)
    result = model.invoke(chat)
    chat.append(result.content)
    print(result.content)