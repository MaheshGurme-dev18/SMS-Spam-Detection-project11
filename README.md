# 📩 SMS Spam Detection using Machine Learning

## 📌 Project Overview

SMS spam detection is a Natural Language Processing (NLP) classification problem that identifies whether a text message is **Spam** or **Ham (Not Spam)**.

This project uses **TF-IDF Vectorization** and the **Multinomial Naive Bayes** algorithm to classify SMS messages with high accuracy. The project includes text preprocessing, feature extraction, model training, evaluation, and model serialization for future deployment.

---

# 🎯 Problem Statement

The objective of this project is to build a Machine Learning model that automatically classifies SMS messages into:

- 📧 Ham (Legitimate Message)
- 🚫 Spam (Unwanted Promotional/Fraudulent Message)

---

# 📊 Dataset

The dataset contains SMS messages with two columns:

| Column | Description |
|---------|-------------|
| label | Spam or Ham |
| message | SMS Text |

### Target Classes

| Label | Meaning |
|--------|----------|
| ham | Legitimate Message |
| spam | Spam Message |

---

# ⚙️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- WordCloud
- Regular Expressions (re)
- String Module
- Scikit-learn
- Joblib

---

# 📚 Libraries Used

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import joblib
```

---

# 📂 Project Structure

```
SMS-Spam-Detection/
│
├── data/
│      └── spam.csv
│
├── notebooks/
│      └── SMS_Spam_Detection.ipynb
│
├── models/
│      ├── spam_classifier.pkl
│      └── tfidf_vectorizer.pkl
│
├── src/
│      ├── preprocessing.py
│      ├── train.py
│      └── predict.py
│
├── app.py
├── requirements.txt
├── README.md
└── assets/
```

---

# 🔍 Exploratory Data Analysis (EDA)

Performed analysis includes:

- Dataset Overview
- Missing Value Detection
- Duplicate Removal
- Class Distribution
- Message Length Analysis
- Spam vs Ham Distribution
- Most Frequent Words
- WordCloud Visualization
- Correlation Analysis

---

# 🧹 Text Preprocessing

The SMS messages were cleaned using the following techniques:

- Convert text to lowercase
- Remove punctuation
- Remove numbers
- Remove special characters
- Remove extra spaces
- Tokenization
- Text normalization

Example:

Before:

```
Congratulations!! You won ₹10,000. Call now!!!
```

After:

```
congratulations won call now
```

---

# 🔠 Feature Engineering

Text data cannot be directly used by Machine Learning algorithms.

Therefore, **TF-IDF (Term Frequency-Inverse Document Frequency)** Vectorization is used to convert text into numerical feature vectors.

Advantages:

- Gives higher importance to informative words
- Reduces the impact of common words
- Produces sparse numerical vectors suitable for classification

---

# 🤖 Machine Learning Algorithm

### Multinomial Naive Bayes

This algorithm is particularly effective for text classification because it:

- Works well with word frequency data
- Is computationally efficient
- Performs well on high-dimensional sparse data
- Provides high accuracy for spam filtering

---

# 🔄 Project Workflow

```
SMS Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Text Preprocessing
      │
      ▼
TF-IDF Vectorization
      │
      ▼
Train-Test Split
      │
      ▼
Model Training
      │
      ▼
Model Evaluation
      │
      ▼
Model Saving
      │
      ▼
Prediction
```

---

# 📈 Model Evaluation

The model is evaluated using:

- Accuracy Score
- Confusion Matrix
- Precision
- Recall
- F1 Score
- Classification Report

---

# 📊 Confusion Matrix

The confusion matrix helps evaluate:

- True Positives
- True Negatives
- False Positives
- False Negatives

---

# 💾 Model Saving

The trained model and TF-IDF vectorizer are saved using **Joblib**.

```python
joblib.dump(model, "spam_classifier.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
```

They can later be loaded without retraining.

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/SMS-Spam-Detection.git
```

Move to project folder

```bash
cd SMS-Spam-Detection
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the project

```bash
python app.py
```

---

# 📦 Requirements

```
pandas
numpy
matplotlib
seaborn
wordcloud
scikit-learn
joblib
```

---

# 📸 Example Predictions

### Example 1

Input

```
Congratulations! You have won a free iPhone. Click the link now.
```

Prediction

```
Spam 🚫
```

---

### Example 2

Input

```
Hey, are we still meeting at 7 PM today?
```

Prediction

```
Ham ✅
```

---

# 📊 Skills Demonstrated

- Data Cleaning
- Exploratory Data Analysis
- Natural Language Processing (NLP)
- Text Preprocessing
- Regular Expressions
- TF-IDF Vectorization
- Machine Learning Classification
- Naive Bayes Algorithm
- Model Evaluation
- Model Serialization
- Python Programming
- Data Visualization

---

# 🔮 Future Improvements

- Add Stopword Removal
- Apply Stemming
- Apply Lemmatization
- Compare multiple ML models (Logistic Regression, SVM, Random Forest)
- Hyperparameter Tuning
- Build a Streamlit Web App
- Deploy using Flask/FastAPI
- Docker Containerization
- Cloud Deployment (AWS/Azure/GCP)
- MLOps Integration
- Real-time SMS Spam Detection API

---

# 📖 Learning Outcomes

Through this project, I learned:

- How NLP differs from traditional Machine Learning.
- Techniques for cleaning and preprocessing text data.
- Converting text into numerical features using TF-IDF.
- Why Multinomial Naive Bayes performs well for text classification.
- Evaluating classification models using Precision, Recall, F1-Score, and Confusion Matrix.
- Saving and reusing trained Machine Learning models with Joblib.

---

# 👨‍💻 Author

**Mahesh Gurme**

**Skills:**

- Data Science
- Machine Learning
- Deep Learning
- Natural Language Processing (NLP)
- Python
- SQL
- Power BI
- AI & Data Analytics

---

# ⭐ If you found this project useful, consider giving it a star on GitHub!
