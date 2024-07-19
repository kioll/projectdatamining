import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer, RobustScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error, 
                             silhouette_score, accuracy_score, f1_score, classification_report)
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import io

# Streamlit App
def main():
    st.title("Data Mining Project")
    st.subheader("Groupe : Enzo Cuoc, Anna Meliz and Jules Gravier")

    st.header("Part I: Initial Data Exploration")

    st.write("The goal of this project is to develop an interactive web application with the help of **Streamlit** to analyze, clean, and visualize data.")

    # Data loading
    st.subheader("Load your CSV data")
    uploaded_file = st.file_uploader("Choose a file", type=["data","csv"])
    
    if uploaded_file is not None:
        delimiter = st.text_input("Enter the delimiter used in your CSV file", value=",")
        data = pd.read_csv(uploaded_file, delimiter=delimiter)
        
        
        st.header("Data Filtering and Selection")
        filter_col = st.selectbox("Select a column to filter", data.columns)
        filter_value = st.text_input(f"Enter value to filter {filter_col}")
        if filter_value:
            data = data[data[filter_col].astype(str).str.contains(filter_value, na=False)]

        st.subheader("Data Preview")
        st.write("First 5 rows of the dataset:")
        st.write(data.head())
        
        st.write("Last 5 rows of the dataset:")
        st.write(data.tail())
        
        st.subheader("Statistical Summary")
        st.write("Number of rows and columns:")
        st.write(data.shape)
        
        st.write("Column names:")
        st.write(data.columns.tolist())
        
        st.write("Number of missing values per column:")
        st.write(data.isnull().sum())
        
        st.write("Basic statistical summary:")
        st.write(data.describe(include='all'))
        
        # Part II: Data Pre-processing and Cleaning
        st.header("Part II: Data Pre-processing and Cleaning")
        
        st.subheader("Handling Missing Values")
        missing_values_option = st.selectbox(
            "Choose a method to handle missing values",
            ("Delete rows with missing values", "Delete columns with missing values",
             "Replace with mean (numeric only)", "Replace with median (numeric only)", 
             "Replace with mode", "KNN Imputation (numeric only)", "Simple Imputation (constant value)")
        )

        data_cleaned = data.copy()

        if missing_values_option == "Delete rows with missing values":
            data_cleaned = data.dropna()
            st.write("Deleted rows with missing values. Remaining data shape:", data_cleaned.shape)
        elif missing_values_option == "Delete columns with missing values":
            data_cleaned = data.dropna(axis=1)
            st.write("Deleted columns with missing values. Remaining data shape:", data_cleaned.shape)
        elif missing_values_option == "Replace with mean (numeric only)":
            num_cols = data.select_dtypes(include=['number']).columns
            imputer = SimpleImputer(strategy='mean')
            data_cleaned[num_cols] = imputer.fit_transform(data[num_cols])
            st.write("Replaced missing values with the mean of each numeric column.")
        elif missing_values_option == "Replace with median (numeric only)":
            num_cols = data.select_dtypes(include=['number']).columns
            imputer = SimpleImputer(strategy='median')
            data_cleaned[num_cols] = imputer.fit_transform(data[num_cols])
            st.write("Replaced missing values with the median of each numeric column.")
        elif missing_values_option == "Replace with mode":
            imputer = SimpleImputer(strategy='most_frequent')
            data_cleaned = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
            st.write("Replaced missing values with the mode of each column.")
        elif missing_values_option == "KNN Imputation (numeric only)":
            num_cols = data.select_dtypes(include=['number']).columns
            imputer = KNNImputer(n_neighbors=5)
            data_cleaned[num_cols] = imputer.fit_transform(data[num_cols])
            st.write("Replaced missing values using KNN imputation for numeric columns.")
        elif missing_values_option == "Simple Imputation (constant value)":
            fill_value = st.text_input("Enter the constant value to replace missing values with", value="0")
            imputer = SimpleImputer(strategy='constant', fill_value=fill_value)
            data_cleaned = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
            st.write(f"Replaced missing values with the constant value: {fill_value}.")
        
        st.write("Data after handling missing values:")
        st.write(data_cleaned.head())

        # Part III: Data Normalization
        st.subheader("Choose a method to normalize the data")
        normalization_option = st.selectbox(
            "Normalization method",
            ("None", "Min-Max Normalization", "Z-score Standardization", "Quantile Transformation", "Robust Scaler")
        )

        if normalization_option != "None":
            num_cols = data_cleaned.select_dtypes(include=['number']).columns
            data_cleaned_numeric = data_cleaned[num_cols]
            
            if normalization_option == "Min-Max Normalization":
                scaler = MinMaxScaler()
                data_cleaned[num_cols] = scaler.fit_transform(data_cleaned_numeric)
                st.write("Applied Min-Max Normalization to numeric columns.")
            elif normalization_option == "Z-score Standardization":
                scaler = StandardScaler()
                data_cleaned[num_cols] = scaler.fit_transform(data_cleaned_numeric)
                st.write("Applied Z-score Standardization to numeric columns.")
            elif normalization_option == "Quantile Transformation":
                scaler = QuantileTransformer(output_distribution='normal')
                data_cleaned[num_cols] = scaler.fit_transform(data_cleaned_numeric)
                st.write("Applied Quantile Transformation to numeric columns.")
            elif normalization_option == "Robust Scaler":
                scaler = RobustScaler()
                data_cleaned[num_cols] = scaler.fit_transform(data_cleaned_numeric)
                st.write("Applied Robust Scaler to numeric columns.")

        st.write("Data after normalization:")
        st.write(data_cleaned.head())

        # Part IV: Data Visualization
        st.header("Visualization of the cleaned data")
        
        column_to_visualize = st.selectbox("Choose a column to visualize", data_cleaned.columns)
        
        if pd.api.types.is_numeric_dtype(data_cleaned[column_to_visualize]):
            st.subheader("Histogram")
            fig, ax = plt.subplots()
            ax.hist(data_cleaned[column_to_visualize].dropna(), bins=30, edgecolor='k')
            ax.set_title(f'Histogram of {column_to_visualize}')
            ax.set_xlabel(column_to_visualize)
            ax.set_ylabel('Frequency')
            st.pyplot(fig)
            
            st.subheader("Box Plot")
            fig, ax = plt.subplots()
            ax.boxplot(data_cleaned[column_to_visualize].dropna())
            ax.set_title(f'Box Plot of {column_to_visualize}')
            ax.set_xlabel(column_to_visualize)
            ax.set_ylabel('Value')
            st.pyplot(fig)
        else:
            st.write(f"The selected column '{column_to_visualize}' is not numeric and cannot be visualized using histograms or box plots.")

        # Part V: Clustering or Prediction
        st.header("Clustering or Prediction")

        task = st.selectbox("Choose a task", ["Clustering", "Prediction"])

        num_cols = data_cleaned.select_dtypes(include=['number']).columns

        if task == "Clustering":
            st.subheader("Clustering Algorithms")
            clustering_algorithm = st.selectbox(
                "Choose a clustering algorithm",
                ("K-means", "DBSCAN")
            )

            if clustering_algorithm == "K-means":
                st.subheader('1.3 Vary the Number of Clusters from 2 to 10 and Measure the Silhouette Index')
                silhouette_scores = []
                for k in range(2, 11):
                    kmeans = KMeans(n_clusters=k, random_state=42).fit(data_cleaned[num_cols])
                    labels = kmeans.labels_
                    silhouette_scores.append(silhouette_score(data_cleaned[num_cols], labels))
                fig, ax = plt.subplots()
                ax.plot(range(2, 11), silhouette_scores, marker='o')
                ax.set_title('Silhouette Scores for Different Numbers of Clusters')
                ax.set_xlabel('Number of clusters')
                ax.set_ylabel('Silhouette Score')
                st.pyplot(fig)
                optimal_clusters = np.argmax(silhouette_scores) + 2
                st.write(f"Optimal number of clusters based on silhouette score: {optimal_clusters}")

                n_clusters = st.slider("Select number of clusters", 2, 10, optimal_clusters)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                data_cleaned['KMeans_Cluster'] = kmeans.fit_predict(data_cleaned[num_cols])
                
                

                st.subheader('PCA Visualization of Clusters')
                pca = PCA(n_components=2)
                pca_result = pca.fit_transform(data_cleaned[num_cols])
                fig, ax = plt.subplots()
                scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], c=data_cleaned['KMeans_Cluster'], cmap='viridis')
                centroids = pca.transform(kmeans.cluster_centers_)
                ax.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, alpha=0.75, marker='X')
                
                legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
                ax.add_artist(legend1)
                ax.set_title('PCA of K-Means Clusters')
                st.pyplot(fig)

            elif clustering_algorithm == "DBSCAN":
                st.subheader('DBSCAN Parameters')
                eps = st.slider("Epsilon (eps)", 0.1, 5.0, 0.5)
                min_samples = st.slider("Minimum samples", 1, 20, 5)

                dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(data_cleaned[num_cols])
                data_cleaned['DBSCAN_Cluster'] = dbscan.labels_

                st.subheader('DBSCAN Clustering Results')
                st.write(data_cleaned[['DBSCAN_Cluster']].value_counts())

                st.subheader('PCA Visualization of DBSCAN Clusters')
                pca = PCA(n_components=2)
                pca_result = pca.fit_transform(data_cleaned[num_cols])
                fig, ax = plt.subplots()
                scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], c=data_cleaned['DBSCAN_Cluster'], cmap='viridis')
                legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
                ax.add_artist(legend1)
                ax.set_title('PCA of DBSCAN Clusters')
                
                st.pyplot(fig)
                
        elif task == "Prediction":
            st.subheader("Choose Target Variable")
            target = st.selectbox("Select the target variable", data_cleaned.columns)
            features = [col for col in data_cleaned.columns if col != target]
            X = data_cleaned[features]
            y = data_cleaned[target]

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            st.subheader("Choose a prediction algorithm")
            algorithm = st.selectbox("Choose an algorithm", ["Random Forest", "Logistic Regression"])

            

            if algorithm == "Random Forest":
                model = RandomForestClassifier() if y.nunique() <= 2 else RandomForestClassifier()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                if y.nunique() <= 2:
                    st.write("Random Forest Classification Performance:")
                    st.write(f"Accuracy Score: {accuracy_score(y_test, y_pred)}")
                   
                

            elif algorithm == "Logistic Regression":
                model = LogisticRegression(max_iter=1000)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                st.write("Logistic Regression Performance:")
                st.write(f"Accuracy Score: {accuracy_score(y_test, y_pred)}")
                
                
if __name__ == "__main__":
    main()
