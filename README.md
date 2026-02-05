# 📉 Customer Churn Prediction using Machine Learning & Streamlit

Customer churn is one of the most critical challenges faced by subscription-based businesses.  
This project predicts whether a customer is **likely to churn or stay** using Machine Learning and provides an **interactive Streamlit web application** for real-time predictions.

---

## 🚀 Project Overview

- Built a **Customer Churn Prediction system** using Logistic Regression
- Designed a **simple & user-friendly Streamlit UI**
- Reduced complex ML inputs to **only 6 important features**
- Focused on **business-oriented evaluation** (Recall for churn customers)

---

## 🎯 Problem Statement

Customer churn occurs when customers stop using a company’s service.  
The goal of this project is to:
- Predict churn in advance
- Help businesses take **preventive retention actions**
- Reduce revenue loss

---

## 🧠 Approach & Workflow

### 1️⃣ Data Collection
- Dataset: `churn.csv`
- Telecom customer data with demographic, service, and billing information

---

### 2️⃣ Data Preprocessing
- Removed irrelevant column: `customerID`
- Encoded categorical features using **Label Encoding**
- Standardized numerical features using **StandardScaler**
- Handled class imbalance using **class_weight='balanced'**

---

### 3️⃣ Feature Selection (UI Simplification)
Although the model was trained on **19 features**, only **6 most important features** are taken as user input in the Streamlit app:

- `tenure`
- `MonthlyCharges`
- `Contract`
- `InternetService`
- `OnlineSecurity`
- `TechSupport`

👉 Remaining features are automatically filled using:
- Mean (for numerical)
- Mode (for categorical)

This improves **usability without affecting model performance**.

---

### 4️⃣ Model Used
- **Logistic Regression**
- Chosen for:
  - Interpretability
  - Fast training
  - Business-friendly decision boundaries

---

### 5️⃣ Model Evaluation

#### 📊 Classification Report

| Class | Precision | Recall | F1-score |
|------|----------|--------|---------|
| Stay (0) | 0.92 | 0.73 | 0.81 |
| Churn (1) | 0.52 | 0.82 | 0.64 |

- **Accuracy:** 75%
- **High Recall (82%) for churn customers**  
  → Ensures fewer churn customers are missed

📌 From a business perspective, **catching churn customers is more important than avoiding false alarms**.

---

## 🧠 Business Interpretation

- Customers with **high monthly charges** and **month-to-month contracts** are more likely to churn
- Long-term contracts and low billing reduce churn probability
- The model prioritizes **recall over precision** for churn detection

---

## 🔮 Streamlit Web Application

### Features:
- Simple and clean UI
- Only 6 inputs required from user
- Real-time churn prediction
- Dataset preview
- Model performance display

### Prediction Output:
- ✅ Customer is likely to **STAY**
- ❌ Customer is likely to **CHURN**

---

## 🛠️ Tech Stack Used

- Python
- Pandas & NumPy
- Scikit-learn
- Streamlit
- Matplotlib & Seaborn

---

## ▶️ How to Run the Project

```bash
pip install -r requirements.txt
streamlit run app.py
