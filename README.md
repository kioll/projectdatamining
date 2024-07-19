# Data Mining Project

## Group: Enzo Cuoc, Anna Meliz, and Jules Gravier

## Project Objective

The goal of this project is to develop an interactive web application using **Streamlit** to analyze, clean, and visualize data. The application should also implement clustering algorithms to group similar objects in the dataset, as well as prediction algorithms for regression or classification problems.

## Prerequisites

Before running the project, ensure you have installed the necessary libraries. You can install them using pip:

```bash
pip install streamlit pandas scikit-learn matplotlib seaborn
```

## Project Structure
The project is divided into four main parts:

1. Initial Data Exploration

2. Data Pre-processing and Cleaning

3. Visualization of the Cleaned Data

4. Clustering or Prediction


## Running the Project
## Step 1: Clone the Repository
### Clone the repository containing the project to your local machine:

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo

```

## Step 2: Create and Activate a Virtual Environment
### Create a virtual environment and activate it:

On macOS and Linux:


```bash
python -m venv myenv
source myenv/bin/activate

```

On Windows:
```bash
python -m venv myenv
myenv\Scripts\activate
```

## Step 3: Install Dependencies
### Install the necessary libraries:

```bash
pip install -r requirements.txt
```

## Step 4: Run the Streamlit Application
### Launch the Streamlit application:

```bash
streamlit run app.py

```

## Using the Application
## Part I: Initial Data Exploration
1. **Load Data:** Upload your CSV file through the application interface.

2. **Data Description:** View a preview of the first and last rows of the dataset, as well as a basic statistical summary.

## Part II: Data Pre-processing and Cleaning
1. **Handling Missing Values:** Choose a method to handle missing values from the following options:

 - Delete rows with missing values
 - Delete columns with missing values
 - Replace with mean (numeric columns only)
 - Replace with median (numeric columns only)
 - Replace with mode
 - KNN Imputation (numeric columns only)
 - Simple Imputation with a constant value

2. **Data Normalization:** Choose a method to normalize the data from the following options:

 - Min-Max Normalization
 - Z-score Standardization
 - Quantile Transformation
 - Robust Scaler
## Part III: Visualization of the Cleaned Data
1. **Histograms:** Visualize the distribution of data for each feature in the form of histograms.

2. **Box Plots:** Visualize the distribution and outliers of each feature using box plots.
## Part IV: Clustering or Prediction
1. **Clustering:** Implement clustering algorithms:

 - K-means
 - DBSCAN
 - Choose the desired algorithm and set its parameters.

2. **Prediction:**  Implement prediction algorithms:

 - Linear Regression
 - Random Forest Classifier
 - Choose the desired algorithm and set its parameters.
 - Select the target column for prediction.

## Conclusion
This project provides an interactive web application for data analysis, cleaning, visualization, clustering, and prediction using Streamlit. Follow the steps above to set up and run the application on your local machine.
