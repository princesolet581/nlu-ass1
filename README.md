# Sports vs Politics Text Classifier

## Overview

This project implements a machine learning based text classification system that automatically categorizes documents into **Sports** or **Politics**. The goal is to study how different machine learning algorithms perform on domain-specific text data.

---

## Objectives

* Build a binary text classifier
* Convert text into numerical features using TF-IDF
* Compare multiple machine learning techniques
* Evaluate performance using accuracy and confusion matrix

---

## Dataset

A small labeled dataset was created containing sentences from two domains:

* **Sports:** cricket, football, Olympics, matches, players
* **Politics:** elections, parliament, government policies, leaders

Each sentence is labeled as either `SPORTS` or `POLITICS`.

---

## Feature Extraction

Text data is converted into vectors using:

**TF-IDF (Term Frequency – Inverse Document Frequency)**

This helps capture the importance of words in documents.

---

## Machine Learning Models Used

1. Naive Bayes
2. Logistic Regression
3. Support Vector Machine (SVM)

---

## Evaluation Metrics

* Accuracy
* Confusion Matrix

These metrics help measure classification performance and misclassification rates.

---

## Results (Sample)

| Model               | Accuracy |
| ------------------- | -------- |
| Naive Bayes         | 83%      |
| Logistic Regression | 100%     |
| SVM                 | 100%     |

SVM and Logistic Regression performed best on the dataset.

---

## How to Run

```bash
pip install scikit-learn
python sports_politics_classifier.py
```

---

## Future Improvements

* Use larger real-world datasets
* Apply deep learning models
* Extend to multi-class news classification

---


