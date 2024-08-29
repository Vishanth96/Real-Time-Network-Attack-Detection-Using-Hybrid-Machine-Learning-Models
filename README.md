

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

* Following data cleaning, I move to feature selection. Feature selection is vital as it helps in choosing the most relevant features that contribute significantly to the prediction task. 
* In this project, features related to network flow, such as flow_duration, fwd_pkts_tot, bwd_pkts_tot, and several others, are selected. 
* The target variable, Attack_type, is also identified, and I separate it from the feature set. 
* This separation allows us to prepare the data for model training by defining our input features (X) and the target (y).

# Step 1.4 : Exploratory Data Analysis (EDA)

* Exploratory Data Analysis (EDA) follows, where I use libraries like Matplotlib and Seaborn to visualize the distribution of key features such as flow_duration.
* Visualizations help in understanding the data distribution and identifying any anomalies or patterns that might influence model training.
* I also plot a correlation matrix to explore the relationships between different features, aiding in further refining our feature selection.

# Step 2: Data Splitting and Model Training


* With the data preprocessed and cleaned, I proceed to split the dataset into training and testing sets.
* This is a critical step to ensure that our models are trained on one portion of the data and tested on another, allowing us to evaluate the model’s performance on unseen data.
* I use an 80-20 split, where 80% of the data is used for training and 20% for testing.
* In this step, I also encode the target labels since they are categorical.
* Label encoding is performed using Scikit-learn’s LabelEncoder, which converts the categorical labels into numeric format suitable for model training.
* I then build and train two different models: a Bidirectional GRU (BiGRU) model and a Bidirectional LSTM (BiLSTM) model. These models are sequentially defined using TensorFlow’s Keras API, with layers such as Bidirectional GRU/LSTM, Dense, and Dropout layers.
* These layers help in capturing the temporal dependencies in the network traffic data, which is crucial for detecting patterns indicative of different types of attacks.
* The models are compiled with the Adam optimizer and trained on the training set.
* After training, the models are evaluated on the test set, and their accuracy and classification reports are printed to assess performance.

# Step 3: Hyperparameter Tuning and Regularization of BiLSTM Model and BiGRU Model

* To further optimize the BiGRU and BiLSTM models, hyperparameter tuning and regularization are performed.
* This step is crucial for improving the model’s performance by finding the best combination of hyperparameters.
* Keras Tuner’s Hyperband method is employed to search for the optimal set of hyperparameters, such as the number of units in GRU/LSTM layers, dropout rates, and learning rates.
* I define functions to build the BiGRU and BiLSTM models, where the hyperparameters are tuned dynamically. Regularization, such as L2 regularization, is also applied to prevent overfitting.
* The tuned models are then evaluated, and the best models are selected based on validation accuracy. These models are saved for future use.

# Step 4: Hybrid Model Development

* Moving beyond standard deep learning models, I develop hybrid models that combine Convolutional Neural Networks (CNNs) with BiGRU and BiLSTM.
* The rationale behind this hybrid approach is to leverage CNN’s ability to capture spatial patterns in the data along with the temporal dependencies captured by GRU/LSTM.
* I define two hybrid models: Hybrid CNN-BiGRU and Hybrid CNN-BiLSTM. The models start with a Conv1D layer that acts as a feature extractor, followed by MaxPooling to reduce the dimensionality.
* The output is then flattened and reshaped to be compatible with the GRU/LSTM layers.
* These hybrid models are trained and evaluated similarly to the previous models, with accuracy and classification reports generated to compare performance.

# Step 5: Random Forest Model Training

* Recognizing the importance of traditional machine learning models, I also train a Random Forest model.
* Random Forest is an ensemble learning method known for its robustness and ability to handle large datasets with many features.
* The dataset is reloaded and processed similarly to the previous steps, with additional feature engineering performed to create new features such as total_packets, pkt_size_ratio, and flow_duration_per_pkt.
* These features enhance the model’s ability to differentiate between normal and malicious traffic.
* The dataset is split into training and testing sets, and a Random Forest model is trained on the training data.
* The model is then evaluated on the test set, with accuracy, classification reports, and confusion matrices generated to assess its performance. The trained Random Forest model is saved for later use.


# More steps will be added soon.

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
