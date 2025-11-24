import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from psycopg_pool import ConnectionPool

from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.postgres import PostgresSaver

# Import the updated tool
from tools.SQL_tool import lookup_university_info

load_dotenv()

# -------------------
# 1. Database Connection (Postgres)
# -------------------
DB_URI = "postgresql://neondb_owner:npg_ldWU2aIuHv9R@ep-green-unit-adoucn1n-pooler.c-2.us-east-1.aws.neon.tech/neondb?sslmode=require"

connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": 0,
}

# -------------------
# 2. LLM & Tools
# -------------------
api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=api_key,
    temperature=0
)

search_tool = DuckDuckGoSearchRun(region="us-en")

tools = [search_tool, lookup_university_info]
llm_with_tools = llm.bind_tools(tools)


# -------------------
# 3. State & Prompt
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# STRICTER PROMPT
SYSTEM_PROMPT = """You are a specialized Career Counsellor. Your knowledge base regarding universities is RESTRICTED to the 'lookup_university_info' tool.

RULES:
1. When a user mentions a university, you MUST call 'get_university_summary'.
2. You DO NOT have internal knowledge about university specifics. You MUST strictly rely on the tool output.
3. If the tool returns "Context found", use that exact text to answer.
4. If the tool returns "No summary found" or an error, you MUST tell the user: "I do not have data on this university in my database." Then, offer to search the web using the search_tool.
5. Do not make up facts.
Always be helpful and encouraging.
"""


def chat_node(state: ChatState):
    """
    LLM node that invokes the model.
    """
    messages = state["messages"]

    if messages:
        print(f"--- Last Message Type: {messages[-1].type} ---")
        if hasattr(messages[-1], 'tool_calls') and messages[-1].tool_calls:
            print(f"--- Previous Tool Call: {messages[-1].tool_calls} ---")

    prompt = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    response = llm_with_tools.invoke(prompt)
    return {"messages": [response]}


pool = ConnectionPool(conninfo=DB_URI, max_size=20, kwargs=connection_kwargs)
checkpointer = PostgresSaver(pool)
checkpointer.setup()

tool_node = ToolNode(tools)

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)


def retrieve_all_threads():
    configs = checkpointer.list(None)
    all_threads = set()
    for config in configs:
        all_threads.add(config.config["configurable"]["thread_id"])
    return list(all_threads)