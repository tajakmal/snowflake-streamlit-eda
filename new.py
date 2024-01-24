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
import numpy as np 

# For data manipulation
import pandas as pd

# For machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score

# For visualization
import matplotlib.pyplot as plt
import seaborn as sns

import io
import sys
from contextlib import redirect_stdout

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
    f = io.StringIO()  # Create a string buffer to capture output
    with redirect_stdout(f):
        # Attempt to execute the user code safely
        try:
            exec(user_code)  # Execute the user provided code
            execution_output = f.getvalue()  # Get the output from the execution
            # Append both code and output to the chat for interactive display
            st.session_state["messages"].append({"role": "user", "content": user_code})
            st.session_state["messages"].append({"role": "assistant", "content": f"Code executed successfully:\n{execution_output}"})
        except Exception as e:
            # Handle exceptions by displaying error messages in the chat
            st.session_state["messages"].append({"role": "user", "content": user_code})
            st.session_state["messages"].append({"role": "assistant", "content": f"Error executing code: {e}"})




# Title of the app
st.title("💬 Chatbot with Snowflake Cortex")

system_message = """You will be acting as an AI Snowflake Snowpark Python Expert. Your goals is to give correct,
executable python code to the user. You are given one table, the table name is {selected_table}.

When you write the python code, remember to always refer to the {selected_table} and make it snowflake focused.
ALWAYS BE SURE TO USE THE CORRECT COLUMN NAMES.
ADDITIONALLY - LOWER CASE ALL COLUMN NAMES IN THE DATAFRAME. Here's an example,
'''
# Fetch Snowpark DataFrame
    df = session.table(selected_table)
    df_columns = df.schema

    # Convert to Pandas DataFrame for visualization
    data = df.to_pandas()

    # Perform EDA using the Pandas DataFrame
    st.subheader("Descriptive Statistics")
    st.dataframe(data.describe())
'''

Here are additional functions related to Snowflake Cortex ONLY USE WHEN ASKED TO.
Example: 
SUMMARIZE CORTEX FUNCTION - USE THIS WHEN ASKED TO
'''
summary_df = df.select(
    call_udf('snowflake.cortex.summarize', df["column_name"]).alias("summary")
)
'''

You don't have to create a new UDF, use this as a template. DO NOT MAKE ANYTHING UP. 
Always access specific columns in a case-insensitive manner. ONLY USE SNOWPARK.
DO NOT HALLUCINATE - DO NOT MAKE ANYTHING UP. DO NOT MAKE UP UDFs
WRITE THE FULL CODE - You will get a bonus of $10000 if you do this correctly.

Think through step by step. Be thorough. Do not write or insert arbitrary code. 

Write python code when needed to provide accurate results: 
"""

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
    # response_df = session.create_dataframe([conversation_history]).select(
    #     call_udf('snowflake.cortex.complete', 'llama2-70b-chat', concat(lit(conversation_history), F.to_varchar(lit(f"\nUser: {prompt}"))))
    # )

    # Call the Snowflake Cortex UDF to get the response
    response_df = session.create_dataframe([f"{system_message}\n{conversation_history}"]).select(
        call_udf('snowflake.cortex.complete', 'llama2-70b-chat', concat(
            lit(f"{system_message}\n{conversation_history}"), 
            F.to_varchar(lit(f"\nUser: {prompt}"))
        ))
    )


    # Collect the response and update the chat
    full_response = response_df.collect()[0][0]
    st.session_state["messages"].append({"role": "assistant", "content": full_response})
    st.chat_message("assistant").write(full_response)