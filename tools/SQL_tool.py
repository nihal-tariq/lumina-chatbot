import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field


load_dotenv(override=True)


DB_URI = os.environ.get('DATABASE_URL')
api_key = os.getenv("GROQ_API_KEY")


class UniversityInput(BaseModel):
    university_name: str = Field(description="The full name of the university to search for (e.g. 'Harvard').")


# --- 2. ATTACH THE SCHEMA TO THE TOOL ---
@tool("lookup_university_info", args_schema=UniversityInput)
def lookup_university_info(university_name: str) -> str:
    """
    Use this tool to find detailed information about a university.
    This tool queries the SQL database to find the summary, url, and timestamp.
    """

    # --- LOGIC REMAINS UNCHANGED BELOW ---

    # --- A. Setup Internal SQL Agent ---

    # 1. Connect to DB
    db = SQLDatabase.from_uri(DB_URI)

    # 2. Setup LLM for the SQL Agent
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=api_key,
        temperature=0
    )

    # 3. Get Tools from Toolkit
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()

    # 4. Specialized System Prompt
    system_prompt = """
    You are an intelligent SQL agent whose ONLY job is to search a PostgreSQL table for university information.

    Schema:
        university (
            id SERIAL PRIMARY KEY,
            uni_name TEXT NOT NULL,
            url TEXT NOT NULL,
            summary TEXT NOT NULL,
            time_stamp TIMESTAMPTZ DEFAULT NOW()
        );

    Your task:
    1. Generate a PostgreSQL SELECT query to find the row where 'uni_name' matches the user's request.
    2. Use ILIKE with wildcards (e.g., '%Name%') to be robust against spelling differences.
    3. Retrieve 'summary' 
    4. Return the summary text as your final answer. 
    5. If no row is found, return the exact string: "No summary found in the database."

    Just return the summary text.
    """

    # 5. Create the ReAct Agent
    # Note: create_agent is legacy in some versions, but we keep it as requested
    agent_executor = create_agent(llm, tools, system_prompt=system_prompt)

    # --- B. Execute the Search ---
    query_message = f"Find the summary for the university named: '{university_name}'"

    try:
        # Invoke the sub-agent
        result = agent_executor.invoke({"messages": [HumanMessage(content=query_message)]})

        # Extract the last message content
        final_response = result["messages"][-1].content
        return final_response

    except Exception as e:
        return f"Error occurred: {str(e)}"