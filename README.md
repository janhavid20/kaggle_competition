# Titanic - Kaggle Machine Learning Competition

This repository contains my solution for the Kaggle competition **“Titanic: Machine Learning from Disaster.”**
The goal of this project is to predict whether a passenger survived the Titanic disaster using machine learning.

---

## 📌 Competition Overview

* **Competition:** Titanic – Machine Learning from Disaster
* **Platform:** Kaggle
* **Objective:** Predict if a passenger survived (1) or not (0)
* **Evaluation Metric:** Accuracy
* **Submission Format:**

  * Columns: `PassengerId`, `Survived`
  * Rows: 418 + header

---

## 📂 Files in This Repository

* `titanic_1.ipynb` → Kaggle notebook with full code
* `submission.csv` → Final prediction file submitted to Kaggle
* `README.md` → Project explanation (this file)

---

## 🧠 Approach

### 1. Data Used

Kaggle provides:

* `train.csv` – labeled data (with Survived column)
* `test.csv` – data for prediction

### 2. Selected Features

We used the following columns:

* `Pclass` – Passenger class
* `Sex` – Gender
* `Age` – Age
* `SibSp` – Siblings/Spouses aboard
* `Parch` – Parents/Children aboard
* `Fare` – Ticket price
* `Embarked` – Port of embarkation

---

## ⚙️ Data Processing

* Filled missing values using median or mode
* Converted text values to numbers:

  * male → 0, female → 1
  * S → 0, C → 1, Q → 2
* Selected only useful features
* Ensured train and test had same columns

---

## 🤖 Model Used

* **Algorithm:** Random Forest Classifier
* **Library:** scikit-learn
* **Reason:** Works well for classification and handles mixed data types

---

## 🔄 Workflow

1. Load data
2. Clean missing values
3. Encode categorical features
4. Select features
5. Train model
6. Predict on test data
7. Create `submission.csv`
8. Submit on Kaggle

---

## 📤 Submission

The final file format:

```
PassengerId,Survived
892,0
893,1
...
```

* Uploaded successfully to Kaggle
* Used for leaderboard scoring

---

## 🎯 Learning Outcomes

* Learned basic ML workflow
* Understood data cleaning and encoding
* Trained a classification model
* Generated Kaggle submission file
* Uploaded project to GitHub as proof

---

## 🔗 Kaggle Competition Link

[https://www.kaggle.com/competitions/titanic](https://www.kaggle.com/competitions/titanic)

---

## ✨ Conclusion

This project demonstrates my first complete machine learning pipeline:
from raw data to model training and competition submission.
It serves as proof of participation and learning in Kaggle competitions.
