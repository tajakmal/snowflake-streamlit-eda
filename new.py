from snowflake.snowpark import Session
import streamlit as st
import pandas as pd
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark.functions import call_udf, col, concat, lit
from snowflake.snowpark import dataframe
from snowflake.snowpark.types import StringType
import snowflake.snowpark.functions as F
from langchain.memory import ConversationBufferMemory  # Import Conversation Buffer Memory
import plotly.express as px
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Read connection parameters from environment variables
connection_parameters = {
    "account": os.getenv("SNOWFLAKE_ACCOUNT"),
    "user": os.getenv("SNOWFLAKE_USER"),
    "password": os.getenv("SNOWFLAKE_PASSWORD"),
    "role": os.getenv("SNOWFLAKE_ROLE"),
    "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
    "database": os.getenv("SNOWFLAKE_DATABASE"),
    "schema": os.getenv("SNOWFLAKE_SCHEMA")
}

# Create a Snowflake Session
session = Session.builder.configs(connection_parameters).create()


# Function to get table names from Snowflake
def get_table_names():
    query = "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = 'PUBLIC'"
    tables_df = session.sql(query).collect()
    return [row['TABLE_NAME'] for row in tables_df]

# Function to fetch data from a selected table without limiting the records
def fetch_data_from_table(table_name):
    sql = f"SELECT * FROM {table_name}"
    data = session.sql(sql).to_pandas()
    return data

# Streamlit Sidebar for Snowflake Tables
st.sidebar.title('Snowflake Data Viewer')
# Table selection dropdown in the sidebar
table_names = ['Select a table'] + get_table_names()
selected_table = st.sidebar.selectbox("Select a table:", table_names)

# Fetch and display data
if selected_table and selected_table != 'Select a table':
    data = fetch_data_from_table(selected_table)
    st.sidebar.write(f"Records from {selected_table}:")
    st.sidebar.dataframe(data)

# Add an Analyze button in the sidebar
if st.sidebar.button('Analyze'):
    if selected_table and selected_table != 'Select a table':
        st.sidebar.write(f"Analysis results for {selected_table}")
        # Implement your analysis logic here
    else:
        st.sidebar.write("Please select a table to analyze.")

# Multi-line text area for code input
user_code = st.sidebar.text_area("Insert your code here", height=300)

# Dropdown for selecting code type (Python or SQL)
code_type = st.sidebar.selectbox("Select code type", ["Python", "SQL"])

if st.sidebar.button('Run Code'):
    # Logic to handle the execution of user-provided code
    # CAUTION: Executing user-provided code can be risky
    st.sidebar.write("Code execution not implemented")




# Title of the app
st.title("💬 Chatbot with Snowflake Cortex")

# Initialize messages in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# Display previous messages
for msg in st.session_state["messages"]:
    role = "user" if msg["role"] == "user" else "assistant"
    st.chat_message(role).write(msg["content"])

# Input for new messages
if prompt := st.chat_input():
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Prepare the conversation history
    conversation_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state["messages"]])

    # Include the selected table in the conversation history
    conversation_history += f"\nSelected Table: {selected_table}"

    # Call the Snowflake Cortex UDF to get the response
    response_df = session.create_dataframe([conversation_history]).select(
        call_udf('snowflake.cortex.complete', 'llama2-70b-chat', concat(lit(conversation_history), F.to_varchar(lit(f"\nUser: {prompt}"))))
    )

    # Collect the response and update the chat
    full_response = response_df.collect()[0][0]
    st.session_state["messages"].append({"role": "assistant", "content": full_response})
    st.chat_message("assistant").write(full_response)