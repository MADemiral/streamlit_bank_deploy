
---

# Bank Marketing Campaign Classification - Machine Learning Model

## Project Overview

This project aims to build a machine learning model to predict whether a client of a bank will subscribe to a term deposit based on a dataset of direct marketing campaigns (phone calls) conducted by a Portuguese bank. The dataset used is the **Bank Marketing Data Set**, specifically the file `bank-additional.csv`, which contains 10% of the examples (4119 instances) and 20 features.

The main objective of the project is to create a model that accurately predicts whether a customer will subscribe to a term deposit based on their personal and campaign-related features.

## Dataset Description

The dataset is from a banking institution's direct marketing campaign, which involved phone calls to clients in an effort to promote term deposits. The goal is to classify whether a customer subscribes to a term deposit (`y` variable), based on the characteristics of the individual and the campaign interactions.

**Features in the dataset include:**
- Client's personal information (e.g., age, job, marital status)
- Campaign-related information (e.g., number of contacts, outcome of the previous campaign)
- Target variable: Whether the client subscribed to a term deposit (`y`)

You can access the dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing).

## Project Steps

The project is divided into several steps, which are outlined below:

### 1. **Data Cleaning**
   - Inspect and clean the dataset for any missing, inconsistent, or incorrect data.
   - Handle any outliers or invalid entries.

### 2. **Data Preprocessing**
   - Perform necessary preprocessing steps like encoding categorical variables and scaling numerical features.
   - Split the dataset into training and testing sets.

### 3. **Feature Selection**
   - Use feature selection techniques such as correlation analysis, mutual information, or Recursive Feature Elimination (RFE) to select the most relevant features.

### 4. **Model Selection**
   - Train and evaluate at least three different machine learning models:
     - Logistic Regression
     - Random Forest
     - Neural Network
   - Choose the best-performing model based on evaluation metrics (accuracy, precision, recall, F1 score).

### 5. **Hyperparameter Tuning**
   - Tune the hyperparameters of the selected model using techniques like Grid Search or Random Search.

### 6. **Evaluation**
   - Evaluate the performance of the final model on the testing dataset using metrics like confusion matrix, precision, recall, and F1 score.

### 7. **Deployment**
   - Deploy the final model using Streamlit.
   - Create a user-friendly web interface where users can input client information and receive a prediction about whether they will subscribe to a term deposit.

## Project Structure

The project contains the following files:

- `model_train/Ada442.ipynb.ipynb`: Jupyter Notebook containing the code for the entire project, including data preprocessing, model building, and evaluation.
- Streamlit deployment link: [Streamlit Project](https://ada442-deploy.streamlit.app)

## Requirements

The following Python libraries are required for the project:

- pandas
- numpy
- scikit-learn
- seaborn
- matplotlib
- streamlit


You can install the necessary libraries using the `requirements.txt` file:

```
pip install -r requirements.txt
```

## Running the Project Locally

To run the project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/MADemiral/streamlit_bank_deploy.git
   cd ADA442/
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Jupyter Notebook**:
   ```bash
   jupyter notebook model_train/Ada442.ipynb
   ```

4. **Deploy the Streamlit app**:
   Run the following command to start the Streamlit app locally:
   ```bash
   streamlit run app.py
   ```

5. **Access the web app**:
   After running the Streamlit app, you can open it in your browser at:
   ```
   http://localhost:8501
   ```

## Deployment on Streamlit Cloud

The model is deployed using Streamlit, and you can interact with the web app through the following link:

**[Streamlit Project Link](https://ada442-deploy.streamlit.app)**

## Data File Location

The dataset used for model training is located in the `model_train/` directory under the file `bank_csv.csv`.


## Conclusion

In this project, we aimed to predict whether a bank customer would subscribe to a term deposit based on direct marketing campaign data. After performing data cleaning, preprocessing, and model selection, we deployed the final model using Streamlit for easy interaction. 

We hope this model can help improve future marketing campaigns and better target potential customers.

---
