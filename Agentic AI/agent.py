from agno.agent import Agent
from agno.tools.yfinance import YFinanceTools
from agno.models.groq import Groq
from dotenv import load_dotenv
from agno.tools.duckduckgo import DuckDuckGoTools

load_dotenv()
agent = Agent(
    model = Groq(id = "qwen/qwen3-32b"),
    tools = [YFinanceTools(),DuckDuckGoTools()],
    add_datetime_to_context =True,
    description = "You are an investment analyst that researches stock prices, analyst recommendations.",
    instructions = ["Use given tools whenever necessary. Format your response using markdown and use tables wherever possible."],
    markdown = True
)

agent.print_response("Share the top stock in India and its analyst recommendations.")