from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import os
from dotenv import load_dotenv
from phi.model.google import Gemini  # Replace with the actual module path

# Load environment variables (for Groq API key, not OpenAI)
load_dotenv()

# Ensure you're using Groq API key and not OpenAI's
groq_api_key = os.getenv("GROQ_API_KEY")  # Make sure to replace it with Groq's API key

# Web search agent with Groq model
web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for the information",
    model=Groq(id="llama-3.2-90b-vision-preview", api_key=groq_api_key),  # Using Groq model here
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tools_calls=True,
    markdown=True,
)

# Financial agent with Groq model
finance_agent = Agent(
    name="Finance AI Agent",
    model=Groq(id="llama-3.2-90b-vision-preview", api_key=groq_api_key),  # Using Groq model here
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_news=True)],
    instructions=["Use tables to display the data"],
    show_tool_calls=True,
    markdown=True,
)

# Multi-agent system
multi_ai_agent = Agent(
model = Gemini(id="models/gemini-2.0-flash-exp"),
team= [web_search_agent, finance_agent],
instructions=["Always include sources", "Use table to display the data"],
show_tools_calls = True,
markdown= True,
)

# Perform the task
multi_ai_agent.print_response("Summarize analyst recommendation and share the latest news for NVDA", stream=True)
