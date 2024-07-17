import streamlit as st
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer

def main():
    st.title("Data Mining Project")
    st.subheader("Groupe : Enzo Cuoc, Anna Meliz and Jules Gravier")

    st.header("Part I: Initial Data Exploration")

    st.write("The goal of this project is to develop an interactive web application with the help of **Streamlit** to analyze, clean and visualize data.")

    # Data loading
    st.subheader("Load your CSV data")
    uploaded_file = st.file_uploader("Choose a file", type=["csv"])
    
    if uploaded_file is not None:
        # Type de séparation (délimiteur)
        delimiter = st.text_input("Enter the delimiter used in your CSV file", value=",")
        
        # Chargement des données
        data = pd.read_csv(uploaded_file, delimiter=delimiter)
        
        # Data description
        st.subheader("Data Preview")
        st.write("First 5 rows of the dataset:")
        st.write(data.head())
        
        st.write("Last 5 rows of the dataset:")
        st.write(data.tail())
        
        # Statistical summary
        st.subheader("Statistical Summary")
        st.write("Number of rows and columns:")
        st.write(data.shape)
        
        st.write("Column names:")
        st.write(data.columns.tolist())
        
        st.write("Number of missing values per column:")
        st.write(data.isnull().sum())
        
        st.write("Basic statistical summary:")
        st.write(data.describe())
        
        # Part II: Data Pre-processing and Cleaning
        st.header("Part II: Data Pre-processing and Cleaning")
        
        # Handling missing values
        st.subheader("Handling Missing Values")
        missing_values_option = st.selectbox(
            "Choose a method to handle missing values",
            ("Delete rows with missing values", "Delete columns with missing values",
             "Replace with mean", "Replace with median", "Replace with mode",
             "KNN Imputation", "Simple Imputation")
        )

        if missing_values_option == "Delete rows with missing values":
            data_cleaned = data.dropna()
        elif missing_values_option == "Delete columns with missing values":
            data_cleaned = data.dropna(axis=1)
        elif missing_values_option == "Replace with mean":
            imputer = SimpleImputer(strategy='mean')
            data_cleaned = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
        elif missing_values_option == "Replace with median":
            imputer = SimpleImputer(strategy='median')
            data_cleaned = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
        elif missing_values_option == "Replace with mode":
            imputer = SimpleImputer(strategy='most_frequent')
            data_cleaned = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
        elif missing_values_option == "KNN Imputation":
            imputer = KNNImputer(n_neighbors=5)
            data_cleaned = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
        elif missing_values_option == "Simple Imputation":
            imputer = SimpleImputer(strategy='constant', fill_value=0)
            data_cleaned = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
        
        st.write("Data after handling missing values:")
        st.write(data_cleaned.head())

if __name__ == "__main__":
    main()
