from agno.agent import Agent
from agno.team import Team
from agno.models.groq import Groq
from dotenv import load_dotenv

load_dotenv()

eng_agent = Agent(
    name = "English Agent",
    role = "You answer questions in English"
)
chinese_agent = Agent(
    name = "Chinese Agent",
    role = "You answer questions in Chinese"
)
hindi_agent = Agent(
    name = "Hindi Agent",
    role = "You answer questions in Hindi"
)


team = Team(
    name = "Answer and Translation Team",
    model = Groq(id = "qwen/qwen3-32b"),
    members = [eng_agent,chinese_agent,hindi_agent],
    markdown = True,
    show_members_responses = True,
    instructions = "All member agents must respond to answer the query in their specific languages. Do not call just one agent. Output the response of all agents."
)

team.print_response("What is the capital of India?")

