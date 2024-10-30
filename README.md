
---

# Breast Cancer Predictor App

![Breast Cancer Predictor WebApp](https://github.com/LearnCode801/End-To-End-Breast-Cancer-Predictor/blob/main/Screenshot%202024-10-30%20135921.png)

## Overview

The **Breast Cancer Predictor App** is an end-to-end application built to predict the presence of breast cancer based on user-provided input parameters. Using a machine learning model trained on the Wisconsin Breast Cancer Dataset, the application provides an interactive interface for users to enter medical details and receive predictions in real-time. This project demonstrates an end-to-end solution, including data preprocessing, model training, and deployment in a user-friendly interface.

## Project Structure

- **Data Preprocessing**: Cleaning and preparing the data for training, handling any null values, and normalizing features.
- **Model Building**: Training machine learning models, such as Logistic Regression, Decision Trees, and K-Nearest Neighbors (KNN).
- **Evaluation**: Model evaluation metrics like accuracy, confusion matrix, and ROC curves are calculated to assess model performance.
- **Deployment**: The selected model is deployed with a web interface built using Flask, allowing users to interact with the model directly.

## Key Features

- **Data Cleaning**: Ensures data is free of errors or inconsistencies.
- **Model Selection**: Provides options for different models with comparisons based on accuracy.
- **Web Application**: User-friendly web interface for non-technical users to interact with the predictive model.
- **Real-time Predictions**: Users can input parameters to get instant predictions.

## Dependencies

- Python 3.x
- Jupyter Notebook
- Flask
- Pandas
- Scikit-Learn

## Installation

Clone the repository and install the dependencies.

```bash
git clone https://github.com/LearnCode801/End-To-End-Breast-Cancer-Predictor.git
cd End-To-End-Breast-Cancer-Predictor
pip install -r requirements.txt
```

## Usage

1. Run the Jupyter Notebook `FSDS_Breast_Cancer_Predictor_App.ipynb` for detailed step-by-step insights into the model training and evaluation process.
2. Start the Flask application for the web interface:

```bash
python app.py
```

3. Navigate to `http://localhost:5000` to access the app in your browser.

## Results

The application predicts the likelihood of breast cancer based on user input parameters, displaying results as "Malignant" or "Benign."

### Confusion Matrix

![Confusion Matrix](https://github.com/LearnCode801/End-To-End-Breast-Cancer-Predictor/blob/main/result.png)

## Contributing

Contributions are welcome! Please submit a pull request with any improvements or updates.

--- 
