# Breast Cancer Predictor App

![Project Demo](https://github.com/LearnCode801/End-To-End-Breast-Cancer-Predictor/blob/main/Screenshot%202024-10-30%20135921.png)

A machine learning application for predicting breast cancer diagnosis using clinical features. This project implements multiple classification algorithms to distinguish between benign and malignant breast tumors, achieving high accuracy for early diagnosis support.

## 📓 Project Notebook

[![View Notebook](https://img.shields.io/badge/View-Jupyter%20Notebook-orange?style=for-the-badge&logo=jupyter)](https://github.com/LearnCode801/End-To-End-Breast-Cancer-Predictor/blob/main/Breast_Cancer_Predictor_App.ipynb)

## 📊 Project Overview

Breast cancer is one of the most common cancers among women worldwide, representing a significant public health challenge. Early diagnosis can dramatically improve prognosis and survival rates by enabling timely clinical treatment. This project aims to accurately classify breast tumors as benign or malignant, helping prevent unnecessary treatments for benign cases.

**Classification Categories:**
- **Benign (0)**: Not likely to develop cancer
- **Malignant (1)**: Likely to develop cancer

## 🎯 Key Features

- **Multiple ML Algorithms**: Implementation of KNN and SVM classifiers
- **High Accuracy**: Achieves up to 98% prediction accuracy
- **Feature Analysis**: Comprehensive correlation and importance analysis
- **Data Preprocessing**: Handles missing values and data type conversions
- **Visualization**: Detailed EDA with correlation heatmaps and distribution plots
- **Model Persistence**: Saved models for deployment using pickle

## 📈 Dataset Information

**Source**: Breast Cancer Wisconsin Dataset  
**Records**: 699 patient samples  
**Features**: 10 clinical attributes  
**Target Variable**: Class (Benign/Malignant)

### Dataset Attributes:
1. **Sample code number**: Patient ID
2. **Clump Thickness**: 1-10 scale
3. **Uniformity of Cell Size**: 1-10 scale
4. **Uniformity of Cell Shape**: 1-10 scale
5. **Marginal Adhesion**: 1-10 scale
6. **Single Epithelial Cell Size**: 1-10 scale
7. **Bare Nuclei**: 1-10 scale
8. **Bland Chromatin**: 1-10 scale
9. **Normal Nucleoli**: 1-10 scale
10. **Mitoses**: 1-10 scale
11. **Class**: 2 (Benign) → 0, 4 (Malignant) → 1

## 🔧 Data Preprocessing

### Data Cleaning Steps:
- **Missing Value Treatment**: Identified 16 missing values (marked as '?') in 'bare_nucleoli' column
- **Imputation Strategy**: Replaced missing values with median (robust to outliers)
- **Data Type Conversion**: Converted all features to appropriate integer types
- **Label Encoding**: Transformed class labels (2→0 for Benign, 4→1 for Malignant)
- **Feature Scaling**: Applied StandardScaler for model optimization

### Data Quality:
- **Original Records**: 699 samples
- **Missing Values**: 16 (2.3% in bare_nucleoli column)
- **Class Distribution**: 
  - Benign (0): 458 samples (65.5%)
  - Malignant (1): 241 samples (34.5%)

## 🤖 Machine Learning Models

### 1. K-Nearest Neighbors (KNN)
- **Configuration**: n_neighbors=5, weights='distance'
- **Accuracy**: 98%
- **Performance**: Excellent classification with distance-weighted voting

### 2. Support Vector Machine (SVM)
- **Configuration**: kernel='linear', C=3, gamma=0.025
- **Accuracy**: 97%
- **Performance**: Strong linear separation of classes

### Model Comparison:
| Algorithm | Accuracy | Precision | Recall | F1-Score |
|-----------|----------|-----------|---------|----------|
| **KNN**   | **98%**  | High      | High    | High     |
| **SVM**   | **97%**  | High      | High    | High     |

## 📊 Feature Analysis

### Feature Importance (Random Forest):
The most important features for classification:
1. **Bare Nuclei**
2. **Size Uniformity** 
3. **Shape Uniformity**
4. **Bland Chromatin**
5. **Clump Thickness**

### Correlation Analysis:
- Strong correlations between size and shape uniformity (0.91)
- High correlation between uniformity features and class prediction
- Most features show right-skewed distributions

## 📊 Model Performance

### Confusion Matrix Results:
**KNN Classifier (98% Accuracy)**
- True Negatives: High accuracy in benign detection
- True Positives: Excellent malignant detection
- Low false positive/negative rates

**SVM Classifier (97% Accuracy)**
- Robust linear separation
- Consistent performance across classes
- Minimal misclassification

### Training Configuration:
- **Train-Test Split**: 70%-30%
- **Cross-Validation**: Applied for model validation
- **Scaling**: StandardScaler normalization
- **Random State**: 42 (for reproducibility)

## 📊 Exploratory Data Analysis

### Key Insights:
1. **Class Distribution**: Dataset is moderately imbalanced (65.5% benign, 34.5% malignant)
2. **Feature Correlations**: Strong correlations between morphological features
3. **Distribution Patterns**: Most features show right-skewed distributions
4. **Missing Data**: Only 2.3% missing values, handled effectively with median imputation

### Visualizations:
- Correlation heatmap showing feature relationships
- Distribution plots for all clinical attributes
- Pairplot analysis revealing class separability
- Feature importance ranking from Random Forest

## 🔮 Clinical Significance

### Medical Impact:
- **Early Detection**: Supports early breast cancer screening
- **Reduced False Positives**: Minimizes unnecessary biopsies and treatments
- **Clinical Decision Support**: Aids healthcare professionals in diagnosis
- **Cost Effective**: Reduces healthcare costs through accurate classification

---

**Note**: This project demonstrates machine learning applications in healthcare. All predictions should be validated by qualified medical professionals.
