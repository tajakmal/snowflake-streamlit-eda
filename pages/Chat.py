from snowflake.snowpark import Session
import streamlit as st
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark.functions import call_udf, col, concat, lit
from snowflake.snowpark import dataframe
from snowflake.snowpark.types import StringType
import snowflake.snowpark.functions as F
from langchain.memory import ConversationBufferMemory  # Import Conversation Buffer Memory
import os
from dotenv import load_dotenv
import sys
import io

import pandas as pd

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

# Add a toggle
on = st.sidebar.toggle('Use RAG', value=False)

# Only show the dropdown if the toggle is on
if on:
    options = ['Default [All Docs]', 'Document Placholder 2', 'Document Placholder 3']  # Replace with your options
    dropdown = st.sidebar.selectbox('Select an option', options)

st.sidebar.header("Code")
# ... existing code ...

# Multi-line text area for code input
user_code = st.sidebar.text_area("Insert your code here", height=300)

# Dropdown for selecting code type (Python or SQL)
code_type = st.sidebar.selectbox("Select code type", ["Python", "SQL"])


if st.sidebar.button('Run Code'):
    if code_type == "Python":
        # Attempt to execute the user code safely
        try:
            # Redirect stdout to a string buffer
            old_stdout = sys.stdout
            sys.stdout = buffer = io.StringIO()

            # Create a dictionary to hold local variables
            local_vars = {}

            # Execute the user provided code
            exec(user_code, {}, local_vars)

            # Restore stdout
            sys.stdout = old_stdout

            # Get the output from the buffer
            output = buffer.getvalue()

            # Check if a DataFrame was created
            df = None
            for var in local_vars.values():
                if isinstance(var, pd.DataFrame):
                    df = var
                    break

            st.session_state["messages"].append({"role": "assistant", "content": "Python code executed successfully."})
            if df is not None:
                # If a DataFrame was created, display it
                st.session_state["messages"].append({"role": "assistant", "content": f"DataFrame:\n{df.to_string()}"})
            else:
                # Otherwise, display the output
                st.session_state["messages"].append({"role": "assistant", "content": f"Output:\n{output}"})
        except Exception as e:
            # Handle exceptions by displaying error messages in the chat
            st.session_state["messages"].append({"role": "assistant", "content": f"Error executing Python code: {e}"})

# ... existing code ...



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

    # Define the model to use
    model = 'llama2-70b-chat'  # You can adjust this based on your requirements or selections
    system_message = '''Answer concisely and accurately: '''

   

    if on:  # If RAG toggle is turned on
        # Query with RAG and chunking
        response_df = session.sql(f'''
        select snowflake.ml.complete(
            '{model}', 
            concat( 
                'Answer the question based on the context. Be concise.','Context: ',
                (select array_agg(*)::varchar from (
                    (select chunk from VECTOR_STORE 
                    order by vector_l2_distance(
                    snowflake.ml.embed_text('e5-base-v2', 
                    '{prompt}'
                    ), chunk_embedding
                    ) limit 5))
                    ),
                'Question: ', 
                '{prompt}',
                'Answer: '
            )
        ) as response;
        ''').to_pandas()
    else:
        # Query without RAG
        response_df = session.sql(f'''
        SELECT SNOWFLAKE.ML.COMPLETE(
            '{model}',concat('{system_message}','{prompt}')
        ) as response
        ''').to_pandas()
        # Call the Snowflake Cortex UDF to get the response
       

    # Collect the response and update the chat
    full_response = response_df['RESPONSE'][0]
    #full_response = response_df.collect()[0][0]
    st.session_state["messages"].append({"role": "assistant", "content": full_response})
    st.chat_message("assistant").write(full_response)