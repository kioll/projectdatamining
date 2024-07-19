import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer, RobustScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, silhouette_score
from sklearn.decomposition import PCA
import numpy as np
import io

def main():
    st.title("Data Mining Project")
    st.subheader("Groupe : Enzo Cuoc, Anna Meliz and Jules Gravier")

    st.header("Part I: Initial Data Exploration")

    st.write("The goal of this project is to develop an interactive web application with the help of **Streamlit** to analyze, clean and visualize data.")

    # Data loading
    st.subheader("Load your CSV data")
    uploaded_file = st.file_uploader("Choose a file", type=["data", "csv"])
    
    if uploaded_file is not None:
        # Type de séparation (délimiteur)
        delimiter = st.text_input("Enter the delimiter used in your CSV file", value=",")
        
        # Chargement des données
        data = pd.read_csv(uploaded_file, delimiter=delimiter)
        
        # Définir num_cols après le chargement des données
        num_cols = data.select_dtypes(include=['number']).columns
        
        # Filtrage et sélection des données
        st.header("Data Filtering and Selection")
        filter_col = st.selectbox("Select a column to filter", data.columns)
        filter_value = st.text_input(f"Enter value to filter {filter_col}")
        if filter_value:
            data = data[data[filter_col].astype(str).str.contains(filter_value, na=False)]
        
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
        st.write(data.describe(include='all'))
        
        # Part II: Data Pre-processing and Cleaning
        st.header("Part II: Data Pre-processing and Cleaning")
        
        # Handling missing values
        st.subheader(" 1. Handling Missing Values")
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
        st.subheader(" 2. Choose a method to normalize the data")
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

            # Redéfinir num_cols après la normalisation
            num_cols = data_cleaned.select_dtypes(include=['number']).columns
        
        st.write("Data after normalization:")
        st.write(data_cleaned.head())

        # Part III: Data Visualization
        st.header("Part III: Visualization of the cleaned data")
        
        # Choose a column for visualization
        column_to_visualize = st.selectbox("Choose a column to visualize", data_cleaned.columns)
        
        # Check if the selected column is numeric
        if pd.api.types.is_numeric_dtype(data_cleaned[column_to_visualize]):
            # Histogram
            st.subheader("Histogram")
            fig, ax = plt.subplots()
            ax.hist(data_cleaned[column_to_visualize].dropna(), bins=30, edgecolor='k')
            ax.set_title(f'Histogram of {column_to_visualize}')
            ax.set_xlabel(column_to_visualize)
            ax.set_ylabel('Frequency')
            st.pyplot(fig)
            
            # Box plot
            st.subheader("Box Plot")
            fig, ax = plt.subplots()
            ax.boxplot(data_cleaned[column_to_visualize].dropna())
            ax.set_title(f'Box Plot of {column_to_visualize}')
            ax.set_xlabel(column_to_visualize)
            ax.set_ylabel('Value')
            st.pyplot(fig)
        else:
            st.write(f"The selected column '{column_to_visualize}' is not numeric and cannot be visualized using histograms or box plots.")

        # Part IV: Clustering or Prediction
        st.header("Part IV: Clustering or Prediction")

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
                kmeans = KMeans(n_clusters=n_clusters)
                data_cleaned['Cluster'] = kmeans.fit_predict(data_cleaned[num_cols])
                st.write(f"Applied K-means clustering with the following number of clusters: {n_clusters}")
                
                # Reorder columns to move 'Cluster' to the beginning
                cluster_col = data_cleaned.pop('Cluster')
                data_cleaned.insert(0, 'Cluster', cluster_col)
                
                st.write(data_cleaned)

                # Apply PCA
                pca = PCA(n_components=2)
                principalComponents = pca.fit_transform(data_cleaned[num_cols])
                pca_df = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])
                pca_df['Cluster'] = data_cleaned['Cluster']

                # Display PCA results
                st.subheader("PCA Results")
                st.write(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")

                # Display PCA components
                st.write("Principal Components:")
                st.write(pd.DataFrame(pca.components_, columns=num_cols, index=['PC1', 'PC2']))
                
                # Scatter plot of clusters
                st.subheader("Cluster Scatter Plot")
                fig, ax = plt.subplots()
                scatter = ax.scatter(pca_df['PC1'], pca_df['PC2'], c=pca_df['Cluster'], cmap='viridis')
                centroids = pca.transform(kmeans.cluster_centers_)
                ax.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, alpha=0.75, marker='X')
                legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
                ax.add_artist(legend1)
                ax.set_xlabel('Principal Component 1')
                ax.set_ylabel('Principal Component 2')
                ax.set_title("Cluster Scatter Plot with Centroids")
                st.pyplot(fig)

            elif clustering_algorithm == "DBSCAN":
                eps = st.slider("Select epsilon value", 0.1, 10.0, 0.5)
                min_samples = st.slider("Select minimum samples", 1, 10, 5)
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                data_cleaned['Cluster'] = dbscan.fit_predict(data_cleaned[num_cols])
                st.write(f"Applied DBSCAN clustering with epsilon: {eps} and minimum samples: {min_samples}")
                
                # Reorder columns to move 'Cluster' to the beginning
                cluster_col = data_cleaned.pop('Cluster')
                data_cleaned.insert(0, 'Cluster', cluster_col)
                
                st.write(data_cleaned)
                
                # Apply PCA
                pca = PCA(n_components=2)
                principalComponents = pca.fit_transform(data_cleaned[num_cols])
                pca_df = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])
                pca_df['Cluster'] = data_cleaned['Cluster']

                # Display PCA results
                st.subheader("PCA Results")
                st.write(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")

                # Display PCA components
                st.write("Principal Components:")
                st.write(pd.DataFrame(pca.components_, columns=num_cols, index=['PC1', 'PC2']))
                
                # Calculate cluster densities
                cluster_counts = pca_df['Cluster'].value_counts().sort_index()
                cluster_densities = cluster_counts / cluster_counts.sum()
                
                # Display cluster densities
                st.write("Cluster densities (proportion of total points):")
                st.write(cluster_densities)
                
                # Scatter plot of clusters
                st.subheader("Cluster Scatter Plot")
                fig, ax = plt.subplots()
                scatter = ax.scatter(pca_df['PC1'], pca_df['PC2'], c=pca_df['Cluster'], cmap='viridis')
                legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
                ax.add_artist(legend1)
                ax.set_xlabel('Principal Component 1')
                ax.set_ylabel('Principal Component 2')
                ax.set_title("Cluster Scatter Plot")

                # Annotate density
                ax.annotate('Density of Clusters:', xy=(1.05, 1.0), xycoords='axes fraction', weight='bold')
                for i, (cluster, density) in enumerate(cluster_densities.items()):
                    ax.annotate(f'Cluster {cluster}: {density:.2%}', xy=(1.05, 0.95 - i*0.05), xycoords='axes fraction')
                
                st.pyplot(fig)

        elif task == "Prediction":
            st.subheader("Prediction Algorithms")
            prediction_algorithm = st.selectbox(
                "Choose a prediction algorithm",
                ("Linear Regression", "Random Forest Classifier")
            )

            target_column = st.selectbox("Choose the target column", data_cleaned.columns)
            
            if pd.api.types.is_numeric_dtype(data_cleaned[target_column]):
                X = data_cleaned[num_cols].drop(columns=[target_column])
                y = data_cleaned[target_column]

                if prediction_algorithm == "Linear Regression":
                    model = LinearRegression()
                    model.fit(X, y)
                    st.write("Fitted a Linear Regression model.")
                    predictions = model.predict(X)
                elif prediction_algorithm == "Random Forest Classifier":
                    model = RandomForestClassifier()
                    model.fit(X, y)
                    st.write("Fitted a Random Forest Classifier model.")
                    predictions = model.predict(X)

                st.write("Model training completed.")

                # Display predictions
                st.subheader("Predictions")
                predictions_df = pd.DataFrame({"Actual": y, "Predicted": predictions})
                st.write(predictions_df.head())

                # Scatter plot of actual vs predicted values
                st.subheader("Actual vs Predicted")
                r2 = r2_score(y, predictions)
                mae = mean_absolute_error(y, predictions)
                rmse = mean_squared_error(y, predictions, squared=False)
                fig, ax = plt.subplots()
                ax.scatter(y, predictions, edgecolors=(0, 0, 0))
                ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
                ax.set_xlabel("Actual")
                ax.set_ylabel("Predicted")
                ax.set_title(f"Actual vs Predicted ({prediction_algorithm})\nR² score: {r2:.2f}\nMAE: {mae:.2f}\nRMSE: {rmse:.2f}")
                st.pyplot(fig)
                # Definitions
                # Definitions
                st.subheader("Definitions of Evaluation Metrics")
                st.markdown("""
                - **R² (coefficient of determination)**: Measures how well the predicted values match the actual values. It ranges from 0 to 1, with higher values indicating better fit.
                - **MAE (Mean Absolute Error)**: Represents the average absolute difference between predicted and actual values. Lower values indicate better fit.
                - **RMSE (Root Mean Squared Error)**: Represents the square root of the average squared difference between predicted and actual values. Lower values indicate better fit.
                """)


            else:
                st.write(f"The target column '{target_column}' is not numeric and cannot be used for regression.")

        # Export cleaned data
        st.subheader("Export Cleaned Data")
        csv = data_cleaned.to_csv(index=False).encode('utf-8')
        st.download_button("Download cleaned data as CSV", data=csv, file_name='cleaned_data.csv', mime='text/csv')

if __name__ == "__main__":
    main()
