import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from psycopg_pool import ConnectionPool

from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.postgres import PostgresSaver


from tools.SQL_tool import lookup_university_info

load_dotenv(override=True)

# 1. Database Connection (Postgres)
DB_URI = os.getenv("DATABASE_URL")

connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": 0,
}

# 2. LLM initialization
api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=api_key,
    temperature=0.5
)

search_tool = DuckDuckGoSearchRun(region="us-en")
search_tool.name = "duckduckgo_search"

tools = [search_tool, lookup_university_info]
llm_with_tools = llm.bind_tools(tools)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


SYSTEM_PROMPT = """You are a compassionate, patient, and knowledgeable Academic & Career Counselor. 
Your goal is to guide students through the stressful process of university selection, admissions, and academic study.

### YOUR PERSONA
- **Tone:** Warm, encouraging, professional, and calm. You are a mentor, not just a search engine.
- **Empathy:** If a student expresses stress, anxiety, or overwhelmed feelings, 
you MUST validate their feelings first before offering solutions. 
Offer distinct study tips or stress-management techniques when appropriate.
- **Scope:** You help with finding university details, understanding admission processes, and providing study techniques.

### TOOL USAGE PROTOCOL (STRICT)
You have access to two tools: `lookup_university_info` (Database) and `duckduckgo_search` (Web).

1. **PRIMARY SOURCE (Database):**
   - When a user asks about a specific university, you MUST FIRST call `lookup_university_info`.
   - **Constraint:** Do not assume you know the data. You must see the tool output to answer.
   - If the tool returns data: Summarize it clearly for the student.

2. **SECONDARY SOURCE (Web Search):**
   - **Trigger:** Only use `duckduckgo_search` if:
     a) The database tool returns "No info found" AND the user specifically asks you to look it up elsewhere.
     b) The user asks for general admission advice or current events not covered by specific university data.
   - **Permission:** If the database is empty, politely inform the user:
    "I don't have that specific university in my internal records. Would you like me to search the web for you?"

### CRITICAL RULES
1. **NO HALLUCINATION:** If you do not have information in the Database or the Web Search result, you must admit it. 
NEVER invent tuition fees, acceptance rates, or deadlines.
2. **Context:** Always use the context provided by the `lookup_university_info` tool as the absolute truth for university details.
3. **Admissions & Study:** Guide them through general admission steps (essays, deadlines, requirements) and 
offer study schedules or habits if they ask for academic help.
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
