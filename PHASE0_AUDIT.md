# Spam Detector v1.0 – Phase 0 Audit Report

## Project Overview

Spam Detector v1.0 is a machine learning-based spam classification system developed using the SMS Spam Collection Dataset. The system classifies incoming messages as either Spam or Ham (Legitimate) using Natural Language Processing (NLP) techniques and a Naive Bayes classifier.

---

## Dataset Analysis

### Dataset Source

SMS Spam Collection Dataset

### Dataset Size

* Total Messages: 5,572
* Ham (Legitimate): 4,825
* Spam: 747

### Class Distribution

| Class | Count | Percentage |
| ----- | ----- | ---------- |
| Ham   | 4,825 | 86.6%      |
| Spam  | 747   | 13.4%      |

### Dataset Characteristics

* Binary Classification Problem
* Moderately Imbalanced Dataset
* Text-based NLP Dataset
* Real-world SMS Messages

### Data Quality Notes

Original dataset contains:

* label
* message
* Unnamed: 2
* Unnamed: 3
* Unnamed: 4

The last three columns contain no useful information and should be removed during preprocessing.

---

## Current Feature Engineering Pipeline

### Text Preprocessing

* Lowercasing
* Tokenization
* Removal of Special Characters
* Removal of Stop Words
* Stemming

### Feature Extraction

Technique:

* TF-IDF Vectorization

Purpose:

* Convert textual messages into numerical vectors suitable for machine learning models.

---

## Current Model Architecture

### Classification Model

Model:

* Multinomial Naive Bayes

Reasons for Selection:

* Fast Training
* Low Computational Cost
* Strong Performance on Text Classification Problems
* Good Baseline for Spam Detection

---

## Model Performance (Baseline)

### Evaluation Metrics

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 97.13% |
| Precision | 99.16% |
| Recall    | 79.19% |
| F1 Score  | 88.06% |

### Interpretation

#### Strengths

* Excellent Accuracy
* Extremely High Precision
* Very Few Legitimate Messages Misclassified as Spam
* Fast Inference Speed

#### Limitations

* Recall is lower than desired
* Approximately 20% of spam messages are missed
* Limited semantic understanding due to TF-IDF representation
* Trained only on static dataset data

---

## Current System Architecture (v1.0)

Dataset (spam.csv)
↓
Text Cleaning
↓
TF-IDF Vectorization
↓
Multinomial Naive Bayes
↓
Prediction
↓
Streamlit User Interface

---

## Explainability

Current Status:

* SHAP Explainability Integrated

Purpose:

* Understand feature importance
* Explain model predictions
* Increase model transparency

---

## Deployment

Current Frontend:

* Streamlit Application

Current Status:

* Local Deployment

---

## Key Limitations of v1.0

1. Uses Static Dataset Only
2. No Real Email Integration
3. No Sender Reputation Analysis
4. No Semantic Embeddings
5. No Model Ensemble
6. No Monitoring System
7. No Automatic Retraining
8. No Production API

---

# Target: Spam Detector v2.0

## Planned Improvements

### Real-Time Data

* Gmail API Integration
* OAuth 2.0 Authentication
* Live Email Fetching

### Advanced NLP

* Sentence Transformers
* Semantic Embeddings
* Linguistic Anomaly Detection

### Advanced Features

* Sender Reputation Scoring
* Domain-Based Analysis
* Historical Behavior Tracking

### Model Improvements

* Random Forest
* XGBoost
* SVM
* Ensemble Stacking

### Explainability

* Interactive SHAP Dashboard
* Prediction Breakdown

### Production Readiness

* FastAPI REST API
* Docker Deployment
* Cloud Hosting

### Monitoring

* Prediction Logging
* Accuracy Tracking
* Data Drift Detection
* Automated Retraining

---

## Baseline vs Target

### v1.0

Static Dataset
→ TF-IDF
→ Naive Bayes
→ Streamlit

### v2.0

Live Gmail Emails
→ Advanced Feature Engineering
→ Embeddings + Reputation Features
→ Ensemble Models
→ SHAP Dashboard
→ FastAPI
→ Monitoring & Retraining
→ Cloud Deployment

---

## Portfolio Statement

Built a machine learning-based Spam Detection System using TF-IDF and Multinomial Naive Bayes on 5,572 real-world SMS messages, achieving 97.13% accuracy and 88.06% F1 score. Currently upgrading the project into a production-grade email spam detection platform with live Gmail integration, advanced NLP embeddings, explainable AI, REST APIs, monitoring, and automated retraining capabilities.
