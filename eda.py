import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.title('Automated EDA Tool')

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file here", type=["csv"])

# Initialize `num_columns` outside the if block to ensure it's accessible later
num_columns = []

if uploaded_file is not None:
    # Reading the uploaded CSV file
    df = pd.read_csv(uploaded_file)
    
    # Displaying the dataset
    st.write("Data Preview:")
    st.write(df.head())
    
    # Storing and displaying the first 5 records in an array format
    first_five_records = df.head().values
    
    # Data Overview
    st.subheader("Data Overview")
    st.write(f"Shape of dataset: {df.shape[0]} rows, {df.shape[1]} columns.")
    st.write("Data Types:")
    st.write(df.dtypes)
    
    # Summary Statistics
    st.subheader("Summary Statistics")
    st.write(df.describe())
    
    # Missing Values
    st.subheader("Missing Values")
    missing_values = df.isnull().sum()
    st.write(missing_values[missing_values > 0])
    
    # Duplicates
    st.subheader("Duplicate Rows")
    st.write(f"Number of duplicate rows: {df.duplicated().sum()}")
    
    # Data Visualizations
    st.subheader("Data Visualizations")
    
    # Update `num_columns` based on the uploaded file
    num_columns = df.select_dtypes(include=np.number).columns.tolist()
    
    # Numerical Columns for Histograms
    if num_columns:
        selected_num_col = st.selectbox('Select Numerical Column for Histogram', options=num_columns)
        sns.histplot(df[selected_num_col].dropna(), kde=True, bins=30)
        st.pyplot(plt)
        plt.clf()  # Clear figure after rendering
        
    # Categorical Columns for Bar Plots
    cat_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_columns:
        selected_cat_col = st.selectbox('Select Categorical Column for Bar Plot', options=cat_columns)
        sns.countplot(y=df[selected_cat_col].dropna())
        plt.xticks(rotation=45)
        st.pyplot(plt)
        plt.clf()  # Clear figure after rendering

    # Ensure `num_columns` is used within a scope where its definition is guaranteed
    if num_columns:
        # Scatter Plot for Numerical Features
        st.subheader("Scatter Plot")
        col1 = st.selectbox('Select the first column for Scatter Plot', options=num_columns, index=0)
        col2 = st.selectbox('Select the second column for Scatter Plot', options=num_columns, index=1 if len(num_columns) > 1 else 0)
        
        if col1 != col2:
            sns.scatterplot(x=df[col1], y=df[col2], hue=df[col2], palette="viridis")
            st.pyplot(plt)
            plt.clf()  # Clear figure after rendering
        else:
            st.warning("Please select two different columns for the scatter plot.")

    