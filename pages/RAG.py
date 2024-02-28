import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
from PyPDF2 import PdfFileReader
from io import BytesIO
from snowflake.snowpark.functions import call_udf, col, concat, lit
from snowflake.snowpark import types as T
from snowflake.snowpark.files import SnowflakeFile
from snowflake.snowpark.types import StringType
from snowflake.snowpark import Session
from snowflake.snowpark.types import StringType, StructField, StructType
import snowflake.snowpark.functions as F
import os
from dotenv import load_dotenv
import tempfile

# Load .env file
load_dotenv()

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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


# Title of the app
st.title("RAG")

# File uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Get the original file name
    original_file_name = uploaded_file.name
    logging.info(f"Processing file: {original_file_name}")

    # Write the uploaded file's content to a new file with the original name
    with open(original_file_name, 'wb') as f:
        f.write(uploaded_file.getvalue())
    logging.info("File written to local system.")
        
    # Upload the new file to the Snowflake stage
    session.file.put(original_file_name, "@RAG", auto_compress=False, overwrite=True)
    logging.info("File uploaded to Snowflake stage.")
    
    # Delete the new file from the local file system
    os.remove(original_file_name)
    logging.info("File deleted from local system.")

    # Process the uploaded file with SNOWPARK_PDF and CHUNK_TEXT
    session.sql('''
    CREATE OR REPLACE TABLE RAW_TEXT AS
    SELECT relative_path, file_url, snowpark_pdf(build_scoped_file_url(@RAG, relative_path)) as raw_text
    FROM directory(@RAG)
    ''').collect()
    logging.info("RAW_TEXT table created.")

    session.sql('''
    CREATE OR REPLACE TABLE CHUNK_TEXT AS
    SELECT
            relative_path,
            func.*
        FROM raw_text AS raw,
             TABLE(chunk_text(raw_text)) as func;
                ''').collect()
    logging.info("CHUNK_TEXT table created.")

    # Insert the results into the VECTOR_STORE table
    session.sql('''
    INSERT INTO VECTOR_STORE (EPISODE_NAME, CHUNK, chunk_embedding)
    SELECT
    RELATIVE_PATH as EPISODE_NAME,
    CHUNK AS CHUNK,
    snowflake.cortex.embed_text('e5-base-v2', chunk) as chunk_embedding
    FROM CHUNK_TEXT
    WHERE RELATIVE_PATH NOT IN (SELECT EPISODE_NAME FROM VECTOR_STORE)
    ''').collect()
    logging.info("Data inserted into VECTOR_STORE table.")
    
# Query the list of files in the stage
result = session.sql("LIST @RAG").collect()

# Convert the result to a pandas DataFrame
df = pd.DataFrame(result, columns=["name", "last_modified", "size", "type"])

# Display the DataFrame in Streamlit
st.header("List in RAG")
#st.write(df)

# Searchable table
search_term = st.text_input("Search")
filtered_df = df[df["name"].str.contains(search_term)]
st.write(filtered_df)