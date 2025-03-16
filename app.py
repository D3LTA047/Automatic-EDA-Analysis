import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(uploaded_file):
    """Loads dataset from an uploaded CSV file."""
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

def handle_missing_values(df):
    """Handles missing values by filling with the median or mode."""
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column].fillna(df[column].mode()[0], inplace=True)
        else:
            df[column].fillna(df[column].median(), inplace=True)
    return df

def categorical_data_analysis(df):
    """Analyzes categorical data by showing unique values and frequencies."""
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        st.write(f"**{col}**: {df[col].nunique()} unique values")
        st.write(df[col].value_counts())

def feature_engineering(df):
    """Encodes categorical features and scales numerical data."""
    le = LabelEncoder()
    scaler = StandardScaler()
    
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = le.fit_transform(df[column])
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

def plot_correlation_matrix(df):
    """Displays the correlation matrix."""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    st.pyplot(fig)

def export_summary(df):
    """Exports a summary report to a CSV file."""
    summary = df.describe().T
    summary.to_csv("eda_summary.csv")
    df.to_csv("processed_data.csv", index=False)
    st.success("Summary and processed dataset saved!")

def main():
    st.title("Automatic EDA Tool")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file:
        df = load_data(uploaded_file)
        if df is not None:
            st.subheader("Dataset Overview")
            st.write(df.head())

            with st.expander("Categorical Data Analysis"):
                categorical_data_analysis(df)

            with st.expander("Missing Values Handling"):
                df = handle_missing_values(df)
                st.write("Missing values filled.")
                st.write(df.isnull().sum())

            with st.expander("Feature Engineering"):
                df = feature_engineering(df)
                st.write("Categorical encoding and scaling applied.")
                st.write(df.head())

            with st.expander("Correlation Matrix"):
                plot_correlation_matrix(df)

            with st.expander("Interactive Scatter Plot"):
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) >= 2:
                    fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], title="Scatter Plot")
                    st.plotly_chart(fig)

            if st.button("Export Reports"):
                export_summary(df)

if __name__ == "__main__":
    main()
