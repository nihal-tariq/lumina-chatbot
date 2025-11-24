import os
from langchain_core.tools import tool
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# 1. DATABASE SETUP
# Replace this with your actual connection string
# Format: postgresql://username:password@host:port/database_name
DB_CONNECTION_STRING = "postgresql://neondb_owner:npg_ldWU2aIuHv9R@ep-green-unit-adoucn1n-pooler.c-2.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"

# Create the engine and session factory
engine = create_engine(DB_CONNECTION_STRING)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@tool
def get_university_summary(university_name: str) -> str:
    """
    Search the database for a specific university by name to retrieve its summary.

    Use this tool whenever a user asks for details, history, or an overview of a
    university. It returns the 'summary' text column which acts as the ground truth
    context for answering the user's question.

    Args:
        university_name (str): The name of the university to search for (e.g., 'Harvard', 'MIT').
    """
    session = SessionLocal()
    try:
        # We use ILIKE for case-insensitive matching
        # We add wildcards (%) to handle cases where user says "Oxford" but DB has "University of Oxford"
        search_term = f"%{university_name}%"

        query = text("""
            SELECT uni_name, summary 
            FROM university 
            WHERE uni_name ILIKE :name 
            LIMIT 1
        """)

        result = session.execute(query, {"name": search_term}).fetchone()

        if result:
            found_name, summary = result
            return f"Context found for {found_name}: {summary}"
        else:
            return f"No summary found in the database for a university matching '{university_name}'."

    except Exception as e:
        return f"Database error occurred while fetching summary: {str(e)}"
    finally:
        session.close()
