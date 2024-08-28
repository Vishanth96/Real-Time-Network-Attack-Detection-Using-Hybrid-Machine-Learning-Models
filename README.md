

# Real-Time Network Attack Detection Using Hybrid Machine Learning Models

## Project Overview
This project focuses on detecting network attacks in real-time by leveraging hybrid machine learning models. The dataset used for this project contains various network traffic features along with labeled attack types. The goal is to preprocess the data, perform feature selection, and then train various machine learning models to accurately classify network traffic and detect potential attacks.

## Dataset
The dataset used in this project is stored in a CSV file named RT_IOT2022.csv. This dataset includes 85 columns with a variety of features related to network traffic. The dataset has been preprocessed to include relevant features for attack detection, and each row is labeled with an attack type.

Sample Dataset Columns:

**id.orig_p**: Source port number

**id.resp_p**: Destination port number

**proto**: Protocol used (e.g., TCP, UDP)

**service**: Service associated with the connection (e.g., MQTT)

**flow_duration**: Duration of the flow in seconds

**fwd_pkts_tot**: Total number of packets sent in the forward direction

**bwd_pkts_tot:** Total number of packets sent in the backward direction

**fwd_data_pkts_tot**: Total number of data packets sent in the forward direction

**bwd_data_pkts_tot**: Total number of data packets sent in the backward direction

**idle.min, idle.max, idle.avg, idle.std**: Idle times and their statistics

**fwd_init_window_size**: Initial window size in the forward direction

**bwd_init_window_size**: Initial window size in the backward direction

**fwd_last_window_size**: Last window size in the forward direction

**Attack_type**: Label indicating the type of attack (e.g., MQTT_Publish)



# Data Loading and Preprocessing

## Step 1.1 : Data Loading

The dataset is loaded from Google Drive into a Pandas DataFrame using the following steps:

```
from google.colab import drive
import pandas as pd
import numpy as np
import sys


drive.mount('/content/drive')


file_path = ('/content/drive/MyDrive/RT_IOT2022.csv')
dataset = pd.read_csv(file_path)


print("First few rows of the dataset:")
print(dataset.head())

```

**Output**: The code snippet above will display the first few rows of the dataset, allowing verification that the data has been loaded correctly.


# Step 1.2 : Data Cleaning
Data cleaning involves identifying and correcting (or removing) inaccuracies and inconsistencies in the data. This step ensures that the dataset is accurate and complete, which is crucial for model reliability. The following activities are typically performed:

1. **Removing Duplicates**: Identify and remove duplicate rows from the dataset.
2. **Correcting Data Types**: Ensure that each feature is of the correct data type (e.g., integers, floats, strings) to avoid errors during analysis.
3. **Outlier Detection and Treatment**: Identify outliers that may skew the results and decide whether to remove or transform them.

# Step 1.3 : Feature Engineering

* Following data cleaning, we move to feature selection. Feature selection is vital as it helps in choosing the most relevant features that contribute significantly to the prediction task. 
* In this project, features related to network flow, such as flow_duration, fwd_pkts_tot, bwd_pkts_tot, and several others, are selected. 
* The target variable, Attack_type, is also identified, and we separate it from the feature set. 
* This separation allows us to prepare the data for model training by defining our input features (X) and the target (y).


# Machine Learning Models
Various machine learning models have been employed to classify network traffic and detect attacks. These models include:

**BiGRU (Bidirectional Gated Recurrent Units)**

**BiLSTM (Bidirectional Long Short-Term Memory)**

**Hybrid CNN-BiGRU (Convolutional Neural Network combined with BiGRU)**

**Hybrid CNN-BiLSTM (Convolutional Neural Network combined with BiLSTM)**

**Random Forest (Ensemble learning method)**


Each model has been trained and evaluated using the preprocessed dataset to determine its effectiveness in detecting different types of network attacks.

# Results
The models are evaluated based on metrics such as accuracy, precision, recall, and F1-score. The performance of each model is compared to identify the most effective approach for real-time network attack detection.


# How to Run the Code
1. **Set up Google Colab**: Ensure that you have access to Google Colab and a Google Drive account.
2. **Upload the Dataset**: Place the RT_IOT2022.csv dataset file in your Google Drive.
3. **Run the Jupyter Notebooks**: The project is divided into several Jupyter notebooks, each corresponding to different steps of the analysis (e.g., Data Loading, Feature Selection, Model Training). Run these notebooks in sequence.
4. **Evaluate Results**: After running the models, review the evaluation metrics to assess the performance.

# Requirements
Python 3.x
Pandas
NumPy
Scikit-learn
TensorFlow / Keras
Google Colab (Optional, for running the notebooks)


# Acknowledgments
This project was developed as part of a research initiative to enhance the detection of network attacks using advanced machine learning techniques.
