import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from snowflake.snowpark import Session
import os
from dotenv import load_dotenv

# Load .env file for Snowflake credentials
load_dotenv()

# Snowflake Connection Parameters
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

# Automated EDA Tool Title
st.title('Automated EDA Tool with Snowflake Data')

# Table selection dropdown
table_names = ['Select a table'] + get_table_names()
selected_table = st.selectbox("Select a table:", table_names)

if selected_table and selected_table != 'Select a table':
    # Fetch data from the selected table
    sql = f"SELECT * FROM {selected_table}"
    df = session.sql(sql).to_pandas()

    # Rest of EDA logic as in your existing code, but using 'df' from Snowflake
    st.write("Data Preview:")
    st.write(df.head())
    
    # Calculating KPIs and formatting them as strings
    dataset_shape = f"{df.shape[0]} rows, {df.shape[1]} columns"
    missing_values = f"{df.isnull().sum().sum()}"  # Total number of missing values as a string
    duplicate_rows = f"{df.duplicated().sum()}"  # Number of duplicate rows as a string
    
    # Create three columns for KPIs
    kpi1, kpi2, kpi3 = st.columns(3)
    
    with kpi1:
        st.subheader("Dataset Shape")
        st.write(dataset_shape)
    
    with kpi2:
        st.subheader("Missing Values")
        st.write(missing_values)
    
    with kpi3:
        st.subheader("Duplicate Rows")
        st.write(duplicate_rows)

    # Create two columns for Data Overview and Summary Statistics
    col1, col2 = st.columns(2)
    
    with col1:
        # Data Overview
        st.subheader("Data Overview")
        st.write(df.dtypes)
    
    with col2:
        # Summary Statistics
        st.subheader("Summary Statistics")
        st.write(df.describe())
    # Missing Values
    st.subheader("Missing Values")
    missing_values = df.isnull().sum()
    st.write(missing_values[missing_values > 0])

    
    # Data Visualizations
    st.subheader("Data Visualizations")

    # Update `cols_per_row` to 2 for two charts per row
    cols_per_row = 2
    
    # Numerical Columns for Histograms
    num_columns = df.select_dtypes(include=np.number).columns.tolist()
    num_rows = (len(num_columns) + cols_per_row - 1) // cols_per_row  # Calculate rows needed
    
    for i in range(num_rows):
        cols = st.columns(cols_per_row)
        for j in range(cols_per_row):
            index = i*cols_per_row + j
            if index < len(num_columns):
                col = num_columns[index]
                with cols[j]:
                    st.write(f"Histogram for {col}")
                    fig, ax = plt.subplots()
                    sns.histplot(df[col].dropna(), kde=True, bins=30, ax=ax)
                    st.pyplot(fig)
                    plt.close(fig)  # Close the figure to free memory

    # Categorical Columns for Bar Plots
    st.subheader("Bar Plots for Categorical Features")
    cat_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    num_rows_cat = (len(cat_columns) + cols_per_row - 1) // cols_per_row  # Calculate rows needed for categorical
    
    for i in range(num_rows_cat):
        cols = st.columns(cols_per_row)
        for j in range(cols_per_row):
            index = i*cols_per_row + j
            if index < len(cat_columns):
                col = cat_columns[index]
                with cols[j]:
                    st.write(f"Bar Plot for {col}")
                    fig, ax = plt.subplots()
                    sns.countplot(y=df[col].dropna(), ax=ax)
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                    plt.close(fig)  # Close the figure to free memory
