# Workplace Communication Bias Detection System

## Overview

This project is an AI-powered Bias Detection and Mitigation System designed to analyze workplace communication and detect potential linguistic bias, toxicity, or stereotypical language in emails or text messages.

The system uses a hybrid machine learning approach by combining:

* Traditional NLP techniques (TF-IDF)
* Semantic similarity analysis
* Rule-based bias detection
* Multiple machine learning models
* Hybrid scoring mechanism

It can also suggest a neutral reformulation of biased text.

---

## Features

* Detect whether text is Biased or Neutral
* Classify Bias Type
* Generate a Neutral / Mitigated Version
* Compare multiple baseline models
* Use an external labeled dataset for domain adaptation
* Generate evaluation graphs automatically
* Provide an interactive user interface using Streamlit

---

## Datasets Used

### 1. Jigsaw Toxic Comment Classification Dataset

Used for training the label-generation model.

Link: https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data

### 2. Enron Email Dataset

Used as the target workplace communication dataset.

Link: https://www.kaggle.com/datasets/wcukierski/enron-email-dataset

---

## Project Pipeline

### 1. External Dataset Training

The Jigsaw Toxic Comment Classification Dataset is used to train a label generation model.

Pipeline:
Jigsaw Dataset → Train Label Model → Label Enron Emails

### 2. Data Preprocessing

Text is cleaned using:

* Lowercasing
* Punctuation removal
* Stopword removal

### 3. Feature Engineering

The final feature vector is created by combining:

* TF-IDF Features
* Semantic Bias Score

Feature Fusion:

Final Feature Vector = TF-IDF + Semantic Score

### 4. Baseline Models

The following machine learning models are trained and compared:

* Logistic Regression
* Support Vector Machine (SVM)
* Naive Bayes
* Random Forest

The best model is selected based on F1 Score.

### 5. Proposed Hybrid Model

The final prediction is made using:

Final Score = 0.6 × Model Probability + 0.2 × Semantic Score + 0.2 × Rule-based Score

This improves detection of subtle workplace bias.

---

## Evaluation Metrics

The system evaluates models using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC Curve
* Precision-Recall Curve
* Confusion Matrix

Saved graphs include:

* ROC curves
* Precision–Recall curves
* Confusion matrices
* Metrics comparison graphs
* Results table image

---

## Project Structure

```bash
project/
│
├── data/
│   ├── emails.csv
│   └── jigsaw_train.csv
│
├── src/
│   ├── preprocessing.py
│   ├── features.py
│   ├── semantic.py
│   ├── mitigation.py
│   ├── bias_type.py
│   ├── models.py
│   └── evaluation.py
│
├── app.py
├── main.py
│
├── best_model.pkl
├── tfidf_vectorizer.pkl
├── scaler.pkl
│
└── README.md
```

---

## Installation

Install dependencies using:

```bash
pip install -r requirements.txt
```

Or manually install:

```bash
pip install streamlit scikit-learn pandas numpy matplotlib seaborn scipy joblib
```

---

## Run Training Pipeline

```bash
python main.py
```

This will:

* Train the Jigsaw label model
* Label the Enron dataset
* Train baseline models
* Evaluate models
* Save graphs
* Save trained model files

---

## Run User Interface

```bash
streamlit run app.py
```

---

## Example Input and Output

Input:

```text
He is too aggressive for leadership.
```

Output:

```text
Bias Detected  
Confidence Score: 0.68  
Bias Type: Stereotype Bias  

Suggested Neutral Version:  
He is assertive in leadership.
```

---

## Future Improvements

* BERT-based bias classification
* SHAP explainability
* Cloud deployment
* Real-time workplace email plugin integration
