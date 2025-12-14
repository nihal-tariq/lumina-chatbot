import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from psycopg_pool import ConnectionPool
import traceback
import json  # Imported for pretty printing arguments if needed

from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.postgres import PostgresSaver

from tools.SQL_tool import lookup_university_info
from tools.web_search import web_search
from helper import load_prompt

load_dotenv(override=True)

# 1. Database Connection
DB_URI = os.getenv("DATABASE_URL")
connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": 0,
}

# 2. LLM initialization
api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=api_key,
    temperature=0.5
)

# Bind the tools
tools = [web_search, lookup_university_info]
llm_with_tools = llm.bind_tools(tools)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


SYSTEM_PROMPT = load_prompt("prompt.txt")


def chat_node(state: ChatState):
    """
    LLM node that invokes the model.
    """
    messages = state["messages"]
    prompt = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    try:
        response = llm_with_tools.invoke(prompt)

        # --- LOGGING START ---
        # Check if the model wants to call any tools
        if hasattr(response, 'tool_calls') and response.tool_calls:
            print("\n" + "=" * 40)
            print("🛑 DETECTED TOOL CALL REQUEST")
            for tc in response.tool_calls:
                tool_name = tc.get('name')
                tool_args = tc.get('args')
                print(f"-> Tool: {tool_name}")
                print(f"-> Args: {json.dumps(tool_args, indent=2)}")
            print("=" * 40 + "\n")
        # --- LOGGING END ---

    except Exception as e:
        print(f"LLM Invocation Error: {e}")
        traceback.print_exc()
        response = AIMessage(
            content="I'm having trouble connecting to my search tools right now. Could you rephrase that?"
        )

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
    if not checkpointer:
        return []
    configs = checkpointer.list(None)
    all_threads = set()
    for config in configs:
        all_threads.add(config.config["configurable"]["thread_id"])
    return list(all_threads)