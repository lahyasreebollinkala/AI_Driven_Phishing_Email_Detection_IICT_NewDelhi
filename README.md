# 🛡️ AI-Driven Phishing Email Detection Using NLP

## 📌 Project Overview

Phishing emails are one of the most common cyber threats used to steal sensitive information such as passwords, banking credentials, and personal data. Traditional rule-based filters often fail to identify sophisticated phishing attacks.

This project presents an Artificial Intelligence-based phishing email detection system that leverages Natural Language Processing (NLP) and Machine Learning techniques to automatically classify emails as **Phishing** or **Legitimate**.

The system performs text preprocessing, feature extraction using TF-IDF, trains multiple machine learning models, evaluates their performance, and provides an interactive Streamlit web application for real-time prediction.

---

# 🎯 Objectives

- Detect phishing emails automatically using NLP.
- Preprocess raw email content for machine learning.
- Extract textual features using TF-IDF Vectorization.
- Train multiple ML algorithms.
- Compare model performance.
- Deploy the best model through a Streamlit web application.
- Improve cybersecurity awareness by providing instant phishing predictions.

---

# 🚀 Features

- Email Text Cleaning
- NLP-based Preprocessing
- TF-IDF Feature Extraction
- Multiple ML Algorithms
- Model Performance Comparison
- Real-time Email Classification
- Interactive Streamlit Interface
- Confidence Score Prediction
- Model Serialization using Pickle
- Easy Deployment

---

# 📂 Project Structure

```
AI_Driven_Phishing_Email_Detection/
│
├── app/
│   ├── app.py
│   ├── templates/
│   └── static/
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── predict.py
│   └── utils.py
│
├── models/
│   ├── phishing_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── label_encoder.pkl
│
├── dataset/
│   ├── raw_dataset.csv
│   └── cleaned_dataset.csv
│
├── notebooks/
│
├── outputs/
│   ├── confusion_matrix.png
│   ├── accuracy_graph.png
│   └── feature_importance.png
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

# 🛠 Technologies Used

## Programming Language

- Python

## Machine Learning

- Scikit-Learn

## Natural Language Processing

- NLTK
- TF-IDF Vectorizer

## Data Processing

- Pandas
- NumPy

## Visualization

- Matplotlib
- Seaborn

## Deployment

- Streamlit

---

# 📊 Dataset

The dataset consists of phishing and legitimate email samples.

Each record contains:

- Email Subject
- Email Body
- Label

Labels:

- 0 → Legitimate Email
- 1 → Phishing Email

---

# ⚙️ Workflow

## Step 1

Collect phishing and legitimate emails.

↓

## Step 2

Clean the text

- Remove HTML
- Remove punctuation
- Lowercase conversion
- Remove stopwords
- Tokenization
- Lemmatization

↓

## Step 3

Convert text into numerical vectors using TF-IDF.

↓

## Step 4

Split dataset

- Training Data
- Testing Data

↓

## Step 5

Train ML Models

- Logistic Regression
- Random Forest
- Naive Bayes
- Neural Network

↓

## Step 6

Evaluate Performance

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

↓

## Step 7

Save Best Model

↓

## Step 8

Deploy using Streamlit

---

# 🤖 Machine Learning Models

## Logistic Regression

Fast and effective linear classifier for binary classification.

## Random Forest

Ensemble learning algorithm using multiple decision trees.

## Naive Bayes

Probabilistic classifier suitable for text classification.

## Neural Network

Learns complex relationships within email content.

---

# 📈 Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

---

# 💻 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/AI-Driven-Phishing-Email-Detection.git
```

Move into project

```bash
cd AI-Driven-Phishing-Email-Detection
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app/app.py
```

---

# ▶️ How to Use

1. Launch the Streamlit application.
2. Enter the email content.
3. Click **Predict**.
4. The model preprocesses the text.
5. TF-IDF converts text into features.
6. The trained model predicts.
7. The result is displayed as:

- Legitimate Email
or
- Phishing Email

along with prediction confidence.

---

# 📷 Sample Output

```
Email Text:

"Your account has been suspended.
Click here immediately to verify your password."

Prediction:

⚠️ PHISHING EMAIL

Confidence:
98.64%
```

---

# 📚 Applications

- Email Security
- Corporate Cybersecurity
- Spam Filtering
- Banking Fraud Prevention
- Educational Demonstrations
- Security Awareness Training

---

# 🔮 Future Enhancements

- Deep Learning (LSTM/BERT)
- URL Reputation Analysis
- Email Attachment Scanning
- Sender Domain Verification
- Browser Extension
- Gmail/Outlook Integration
- Real-time Threat Intelligence
- Multi-language Email Detection

---

# 👨‍💻 Developed By

**Lahya Sree Bollinkala**

B.Tech Computer Science and Engineering

GitHub: https://github.com/lahyasreebollinkala

LinkedIn: https://linkedin.com/in/lahyasreebollinkala

project_URL: [https://aifakenewspredictioniictnewdelhi-mvwgckvhvrymudzsmu5vbr.streamlit.app/](https://aidrivenphishingemaildetectioniictnewdelhi-donszrskpyhwb9g2e4l.streamlit.app/)
---

# 📜 License

This project is developed for educational and research purposes.

---

# ⭐ Acknowledgements

- Indian Institute of Computing and Technology (IICT)
- Scikit-Learn
- NLTK
- Streamlit
- Kaggle Email Dataset
- Python Community
