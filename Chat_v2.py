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

import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Now, you can add logging statements in your code
# For example, before executing the SQL query:
logging.info("Executing SQL query to fetch aggregated chunks.")

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

logging.info("Read connection parameters from environment variables.")

# Create a Snowflake Session
session = Session.builder.configs(connection_parameters).create()
logging.info("Snowflake session created.")

# Add a toggle
on = st.sidebar.toggle('Use RAG', value=False)
logging.info(f"RAG toggle set to: {on}")

# Only show the dropdown if the toggle is on
if on:
    options = ['Default [All Docs]', 'Document Placeholder 2', 'Document Placeholder 3']  # Replace with your options
    dropdown = st.sidebar.selectbox('Select an option', options)
    logging.info(f"Dropdown selection: {dropdown}")

# Function to get table names from Snowflake
def get_table_names():
    logging.info("Fetching table names from Snowflake.")
    query = "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = 'PUBLIC'"
    tables_df = session.sql(query).collect()
    return [row['TABLE_NAME'] for row in tables_df]

# Function to fetch data from a selected table without limiting the records
def fetch_data_from_table(table_name):
    logging.info(f"Fetching data from table: {table_name}")
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
    logging.info(f"Selected table: {selected_table}")
    data = fetch_data_from_table(selected_table)
    st.sidebar.write(f"Records from {selected_table}:")
    st.sidebar.dataframe(data)

# Add an Analyze button in the sidebar
if st.sidebar.button('Analyze'):
    logging.info("Analyze button clicked.")
    if selected_table and selected_table != 'Select a table':
        st.sidebar.write(f"Analysis results for {selected_table}")
        logging.info(f"Analysis started for table: {selected_table}")
        # Implement your analysis logic here
    else:
        st.sidebar.write("Please select a table to analyze.")
        logging.warning("Analyze button clicked without selecting a table.")

# Multi-line text area for code input
user_code = st.sidebar.text_area("Insert your code here", height=300)

# Dropdown for selecting code type (Python or SQL)
code_type = st.sidebar.selectbox("Select code type", ["Python", "SQL"])

if st.sidebar.button('Run Code'):
    logging.info(f"Run Code button clicked with code type: {code_type}")
    if code_type == "Python":
        f = io.StringIO()  # Create a string buffer to capture output
        with redirect_stdout(f):
            # Attempt to execute the user code safely
            try:
                exec(user_code)  # Execute the user provided code
                execution_output = f.getvalue()  # Get the output from the execution
                logging.info("Python code executed successfully.")
                # Append both code and output to the chat for interactive display
                st.session_state["messages"].append({"role": "user", "content": user_code})
                st.session_state["messages"].append({"role": "assistant", "content": f"Code executed successfully:\n{execution_output}"})
            except Exception as e:
                # Handle exceptions by displaying error messages in the chat
                logging.error(f"Error executing Python code: {e}")
                st.session_state["messages"].append({"role": "user", "content": user_code})
                st.session_state["messages"].append({"role": "assistant", "content": f"Error executing code: {e}"})
    else:
        # For SQL or other code types, you can add similar execution and logging logic
        logging.warning("Unsupported code type selected for execution.")

# Title of the app
st.title("💬 Chatbot with Snowflake Cortex")
logging.info("Chatbot with Snowflake Cortex app started.")

# Define the model to use
model = 'llama2-70b-chat'  # You can adjust this based on your requirements or selections
system_message = '''Answer concisely and accurately: '''
rag_system_message = '''Answer the question based on the context. Be concise.Context: {embeddings} 
iGNORE EVERYTHING ELSE'''

logging.info(f"Model set to: {model}")

# Initialize messages in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
    logging.info("Initialized chat messages in session state.")

# Display previous messages
for msg in st.session_state["messages"]:
    role = "user" if msg["role"] == "user" else "assistant"
    st.chat_message(role).write(msg["content"])

# Input for new messages
if prompt := st.chat_input():
    logging.info("New user message received.")
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Prepare the conversation history
    conversation_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state["messages"]])
    logging.info(f"Conversation history: \n\n{conversation_history}\n")

    if selected_table and selected_table != 'Select a table':
        conversation_history += f"\nSelected Table: {selected_table}"
        #table_summary = 
        logging.info(f"Conversation history: \n\n{conversation_history}\n")


    if on:
        # RAG
        logging.info("Pulling information from RAG")
        embeddings = session.sql(f'''
        select array_agg(*)::varchar from (
                        (select chunk from VECTOR_STORE 
                        order by vector_l2_distance(
                        snowflake.cortex.embed_text('e5-base-v2', 
                        '{prompt}'
                        ), chunk_embedding
                        ) limit 5))
        ''').collect()
        logging.info(f"RAG query: print({embeddings})")
        response_df = session.create_dataframe([f"{rag_system_message}\n{embeddings}\n{conversation_history}"]).select(
            call_udf('snowflake.cortex.complete', 'llama2-70b-chat', concat(
                lit(f"{rag_system_message}\n{embeddings}\n{conversation_history}"), 
                F.to_varchar(lit(f"\nUser: {prompt}"))
            ))
        )
        logging.info(print(f"DATAFRAME \n {response_df.show()}"))
    else:
        # Call the Snowflake Cortex UDF to get the response
        logging.info("Calling Snowflake Cortex UDF for response.")
        response_df = session.create_dataframe([f"{system_message}\n{conversation_history}"]).select(
            call_udf('snowflake.cortex.complete', 'llama2-70b-chat', concat(
                lit(f"{system_message}\n{conversation_history}"), 
                F.to_varchar(lit(f"\nUser: {prompt}"))
            ))
        )
        logging.info(print(f"DATAFRAME \n {response_df.show()}"))

    # Collect the response and update the chat
    full_response = response_df.collect()[0][0]
    logging.info("Response received from Snowflake Cortex UDF.")
    st.session_state["messages"].append({"role": "assistant", "content": full_response})
    st.chat_message("assistant").write(full_response)