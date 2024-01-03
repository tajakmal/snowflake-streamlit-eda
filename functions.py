import io
import pandas as pd
import streamlit as st

def df_info(df):
    # Ensure the column names do not have spaces for easier processing
    df.columns = df.columns.str.replace(' ', '_')
    
    # Initialize lists to store column information
    names = []
    nn_count = []
    dtype = []
    unique_non_null_count = []

    # Iterate through each column to gather information
    for col in df.columns:
        names.append(col)
        nn_count.append(df[col].notnull().sum())
        dtype.append(df[col].dtype)
        unique_non_null_count.append(df[col].nunique())

    # Creating DataFrame to display the information
    df_info_dataframe = pd.DataFrame({
        'Column': names, 
        'Non-Null Count': nn_count, 
        'Unique Non-Null Count': unique_non_null_count, 
        'Data Type': dtype
    })

    return df_info_dataframe

def df_isnull(df):
    res = pd.DataFrame(df.isnull().sum()).reset_index()
    res['Percentage'] = round(res[0] / df.shape[0] * 100, 2)
    res['Percentage'] = res['Percentage'].astype(str) + '%'
    return res.rename(columns = {'index':'Column', 0:'Number of null values'})

def number_of_outliers(df):
    
    df = df.select_dtypes(exclude = 'object')
    
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    
    ans = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()
    df = pd.DataFrame(ans).reset_index().rename(columns = {'index':'column', 0:'count_of_outliers'})
    return df

def space(num_lines=1):
    for _ in range(num_lines):
        st.write("")

def sidebar_space(num_lines=1):
    for _ in range(num_lines):
        st.sidebar.write("")


def sidebar_multiselect_container(massage, arr, key):
    
    container = st.sidebar.container()
    select_all_button = st.sidebar.checkbox("Select all for " + key + " plots")
    if select_all_button:
        selected_num_cols = container.multiselect(massage, arr, default = list(arr))
    else:
        selected_num_cols = container.multiselect(massage, arr, default = arr[0])

    return selected_num_cols    
