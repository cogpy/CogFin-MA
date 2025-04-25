import os
from typing import List, Dict, Any, Optional, Union # Added Union
from langchain_ollama import OllamaLLM
from langchain.agents import AgentExecutor, create_react_agent
# Import necessary components for formatting scratchpad
from langchain.agents.format_scratchpad import format_log_to_messages
from langchain.agents.output_parsers import ReActSingleInputOutputParser

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate # Added PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from langchain.callbacks import StdOutCallbackHandler
from pydantic import BaseModel, Field
from langchain.schema import AgentAction, AgentFinish, OutputParserException, AIMessage, HumanMessage # Added AIMessage, HumanMessage
import re

# Import MoneyControl API
try:
    from moneycontrol import moneycontrol_api as mc
except ImportError:
    print("Error: 'moneycontrol' package not found or moneycontrol_api module missing.")
    print("Please ensure you have installed it: pip install moneycontrol-api")
    exit()

# --- Tool Functions ---
# Ensure all functions have proper try-except blocks and return strings

def search_stocks(query: str) -> str:
    """
    Search for stocks on MoneyControl based on the given query.
    Use this first to find the stock ID if you don't have it.
    Input should be the company name or search term.
    """
    print(f"--- Executing SearchStocks Tool with query: {query} ---")
    try:
        results = mc.search.get_search_results(query)
        if not results:
            return "No stocks found matching the query."

        formatted_results = []
        for idx, stock in enumerate(results[:5]):  # Limit results
            name = stock.get('name', 'Unknown')
            mcid = stock.get('mcid', 'Unknown')
            entity_type = stock.get('entity_type', 'Unknown')
            formatted_results.append(f"{idx+1}. Name: {name}, ID: {mcid}, Type: {entity_type}")

        return "Found stocks:\n" + "\n".join(formatted_results)
    except Exception as e:
        return f"Error searching for stocks: {str(e)}"

def get_stock_details(stock_id: str) -> str:
    """
    Get detailed information about a specific stock using its MoneyControl ID (mcid).
    Input must be the stock ID (e.g., 'RI').
    """
    print(f"--- Executing GetStockDetails Tool with stock_id: {stock_id} ---")
    try:
        info = mc.stocks.get_stock_info(stock_id)
        if not info:
            return f"No details found for stock ID: {stock_id}"

        details = [
            f"Name: {info.get('name', 'N/A')}",
            f"Sector: {info.get('sector', 'N/A')}",
            f"Industry: {info.get('industry', 'N/A')}",
            f"Market Cap: {info.get('market_cap', 'N/A')}",
            f"About: {info.get('about', 'No description available')[:200]}..." # Truncate description
        ]
        return "\n".join(details)
    except Exception as e:
        return f"Error getting stock details for ID {stock_id}: {str(e)}"

def get_stock_price(stock_id: str) -> str:
    """
    Get the current price and related information for a stock using its MoneyControl ID (mcid).
    Input must be the stock ID (e.g., 'RI').
    """
    print(f"--- Executing GetStockPrice Tool with stock_id: {stock_id} ---")
    try:
        price_info = mc.stocks.get_stock_price(stock_id)
        if not price_info:
            return f"No price information found for stock ID: {stock_id}"

        price_details = [
            f"Stock ID: {stock_id}",
            f"Current Price: {price_info.get('price', 'N/A')}",
            f"Change: {price_info.get('change', 'N/A')} ({price_info.get('change_percent', 'N/A')}%)",
            f"Day High: {price_info.get('high', 'N/A')}",
            f"Day Low: {price_info.get('low', 'N/A')}",
            f"52-week High: {price_info.get('52_week_high', 'N/A')}",
            f"52-week Low: {price_info.get('52_week_low', 'N/A')}",
            f"Volume: {price_info.get('volume', 'N/A')}"
        ]
        return "\n".join(price_details)
    except Exception as e:
        return f"Error getting stock price for ID {stock_id}: {str(e)}"

def get_mutual_fund_details(fund_id: str) -> str:
    """
    Get detailed information about a specific mutual fund using its MoneyControl ID (mcid).
    Input must be the mutual fund ID.
    """
    print(f"--- Executing GetMutualFundDetails Tool with fund_id: {fund_id} ---")
    try:
        fund_info = mc.mutual_funds.get_mf_details(fund_id)
        if not fund_info:
            return f"No details found for mutual fund ID: {fund_id}"

        fund_details = [
            f"Name: {fund_info.get('name', 'N/A')}",
            f"Category: {fund_info.get('category', 'N/A')}",
            f"NAV: {fund_info.get('nav', 'N/A')}",
            f"AUM: {fund_info.get('aum', 'N/A')}",
            f"Risk: {fund_info.get('risk', 'N/A')}",
            f"Return (1Y): {fund_info.get('return_1y', 'N/A')}%"
        ]
        return "\n".join(fund_details)
    except Exception as e:
        return f"Error getting mutual fund details for ID {fund_id}: {str(e)}"

def get_market_indices() -> str:
    """
    Get information about major market indices like NIFTY, SENSEX, etc.
    Takes no input.
    """
    print(f"--- Executing GetMarketIndices Tool ---")
    try:
        indices_info = mc.indices.get_indices()
        if not indices_info:
            return "No market indices information available."

        index_details = []
        for idx, index in enumerate(indices_info[:8]): # Limit results
            name = index.get('name', 'Unknown')
            current = index.get('current', 'N/A')
            change = index.get('change', 'N/A')
            change_percent = index.get('change_percent', 'N/A')
            index_details.append(
                f"{idx+1}. {name}: {current} ({change} / {change_percent}%)"
            )
        return "Current Market Indices:\n" + "\n".join(index_details)
    except Exception as e:
        return f"Error getting market indices: {str(e)}"


# --- Define Tools for the Agent ---
tools = [
    Tool(
        name="SearchStocks",
        func=search_stocks,
        description="Search for stocks by name or keyword to find their MoneyControl ID (mcid). Example query: 'Reliance Industries'"
    ),
    Tool(
        name="GetStockDetails",
        func=get_stock_details,
        description="Get detailed information (sector, industry, market cap, about) for a stock using its MoneyControl ID (mcid). Example ID: 'RI'"
    ),
    Tool(
        name="GetStockPrice",
        func=get_stock_price,
        description="Get current price information (price, change, high, low, volume) for a stock using its MoneyControl ID (mcid). Example ID: 'RI'"
    ),
    Tool(
        name="GetMutualFundDetails",
        func=get_mutual_fund_details,
        description="Get detailed information (NAV, AUM, risk, returns) for a mutual fund using its MoneyControl ID (mcid)."
    ),
    Tool(
        name="GetMarketIndices",
        func=get_market_indices,
        description="Get current values and changes for major Indian market indices like NIFTY 50 and SENSEX."
    )
]

# --- Create the Agent ---
def create_financial_agent():
    # Initialize the Ollama model
    llm = OllamaLLM(model="deepseek-r1", temperature=0.1, callbacks=[StdOutCallbackHandler()])

    # Define the prompt template WITH explicit ReAct format instructions
    # This template is used by create_react_agent internally
    # Based on langchain.agents.react.agent.create_react_agent prompt structure
    template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}""" # Note: agent_scratchpad is directly appended here

    # Use PromptTemplate for the core ReAct logic
    prompt = PromptTemplate.from_template(template)

    # Define how intermediate steps are formatted into the agent_scratchpad
    # This function converts (AgentAction, Observation) pairs into AIMessage and HumanMessage
    # which is the expected format for the scratchpad in newer LangChain versions
    def format_scratchpad(intermediate_steps: List[tuple[AgentAction, str]]) -> List[Union[AIMessage, HumanMessage]]:
        log = []
        for action, observation in intermediate_steps:
            # Append the LLM's thought process (action log) as an AIMessage
            log.append(AIMessage(content=action.log))
            # Append the tool's output (observation) as a HumanMessage
            log.append(HumanMessage(content=f"Observation: {observation}"))
        return log

    # Create the agent runnable using the standard function
    # This combines the LLM, prompt, and a default output parser
    agent = create_react_agent(llm, tools, prompt)

    # Set up memory (remains the same)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # Create the agent executor
    # We are using the agent created by create_react_agent, which includes its own parser
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        # Use the dedicated function to format the scratchpad correctly
        # This might not be directly supported in all versions, but let's try structuring it this way
        # If this specific argument doesn't exist, the principle is that the executor
        # should use this logic internally. We rely on AgentExecutor's default handling here.
        # agent_scratchpad_formatter=format_scratchpad, # This argument might not exist, relying on default behavior
        handle_parsing_errors="Check your output and make sure it conforms to the expected format. Ensure you use the 'Action:' and 'Action Input:' format correctly, or provide a 'Final Answer:'.",
        max_iterations=5
    )

    # Manually bind the scratchpad formatter if the argument doesn't exist
    # This is a more explicit way to ensure the formatting happens
    # Note: This structure might need adjustment based on specific LangChain versions
    # agent_executor.agent.runnable = agent_executor.agent.runnable | RunnableLambda(
    #     lambda x: {"agent_scratchpad": format_scratchpad(x["intermediate_steps"])}
    # )


    return agent_executor

# --- Run the Agent ---
if __name__ == "__main__":
    print("Creating financial agent...")
    financial_agent = create_financial_agent()
    print("Agent created. Type 'exit' to quit.")

    while True:
        user_input = input("\nAsk a financial question: ")
        if user_input.lower() == 'exit':
            break
        try:
            response = financial_agent.invoke({"input": user_input})
            print("\nFinal Response:")
            if response and "output" in response:
                 print(response["output"])
            else:
                 print("Agent did not produce a final output.")
                 print("Full response for debugging:", response)
        except OutputParserException as ope:
             print(f"\nOutput Parsing Error after retry: The LLM output could still not be understood.")
             print(f"Details: {ope}")
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            import traceback
            traceback.print_exc()
