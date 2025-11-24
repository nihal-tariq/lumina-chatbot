import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from psycopg_pool import ConnectionPool


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.postgres import PostgresSaver


from tools.SQL_tool import get_university_summary

load_dotenv()

# -------------------
# 1. Database Connection (Postgres)
# -------------------
DB_URI = "postgresql://neondb_owner:npg_ldWU2aIuHv9R@ep-green-unit-adoucn1n-pooler.c-2.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"

connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": 0,
}
pool = ConnectionPool(conninfo=DB_URI, max_size=20, kwargs=connection_kwargs)

checkpointer = PostgresSaver(pool)
checkpointer.setup()

# -------------------
# 2. LLM & Tools
# -------------------
api_key = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=api_key,
    temperature=0
)

search_tool = DuckDuckGoSearchRun(region="us-en")

tools = [search_tool, get_university_summary]
llm_with_tools = llm.bind_tools(tools)


class ChatState(TypedDict):
    # Simplified state: 'messages' handles all history, including tool outputs (summaries)
    messages: Annotated[list[BaseMessage], add_messages]


SYSTEM_PROMPT = """You are a dedicated Career Counsellor helping students navigate their higher education options.

Your Responsibilities:
1. Help students find information on relevant universities.
2. When a student mentions a university, you MUST use the 'get_university_summary' tool to search the database for that university.
3. Use the summary returned by the tool as your ONLY ground truth context to answer questions about that university.
4. If the tool returns no summary or says "No summary found", you must explicitly state that you don't have information 
on that university. ask user to search the internet and use the "serach_tool" to get info on that specific university.
5. DO NOT hallucinate or invent details. If it's not in the retrieved context, do not say it.
"""


def chat_node(state: ChatState):
    """
    LLM node that invokes the model.
    We prepend the System Prompt to the message history for every turn
    to ensure the behavior instructions are strictly followed.
    """
    messages = state["messages"]

    # Prepend system message to the current history
    # This instructs the LLM without persisting the system prompt repeatedly in the database
    prompt = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    response = llm_with_tools.invoke(prompt)
    return {"messages": [response]}


tool_node = ToolNode(tools)


graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

# Start -> Chat
graph.add_edge(START, "chat_node")

# Chat -> (Tools OR End)
graph.add_conditional_edges(
    "chat_node",
    tools_condition
)

# Tools -> Chat (Standard loop: Model -> Tool -> Model reads Tool Output -> Answer)
graph.add_edge("tools", "chat_node")

# Compile with Postgres Checkpointer
chatbot = graph.compile(checkpointer=checkpointer)


def retrieve_all_threads():
    configs = checkpointer.list(None)
    all_threads = set()
    for config in configs:
        all_threads.add(config.config["configurable"]["thread_id"])
    return list(all_threads)