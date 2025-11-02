# 💡 Customer Churn Prediction Dashboard — Future Interns Project

This project was built as part of my **Future Interns** internship, where the task was to create a **dashboard that predicts customer churn** based on historical banking/customer data.

It combines **Python machine learning** and **Streamlit visualization** to help businesses identify at-risk customers and take proactive retention actions.

---

During my **Future Interns** internship, I explored the world of **data science and machine learning**, specifically focusing on **customer analytics**.  

This project helped me:  
- Understand how **real-world customer data** can be transformed into actionable insights.  
- Gain hands-on experience with **feature engineering**, preprocessing, and model evaluation.  
- Learn how to build **interactive dashboards** that make ML predictions **accessible to non-technical users**.  
- Improve my skills in **Python, Streamlit, and data visualization**.  

This experience not only strengthened my technical knowledge but also gave me a deeper understanding of how **machine learning can support business decisions**, especially in customer retention and growth strategies.  

It was a journey of connecting theory to practice, from raw data to a fully functional predictive dashboard.

---

## 🚀 Project Overview

**Goal:**  
Build an interactive system that predicts whether a customer is likely to churn, and visualize the prediction along with feature importance.

**Key Features:**
- Predict customer churn using a **trained machine learning model**  
- Interactive sidebar to input customer features  
- Display predicted probability and classification (`Churned` / `Retain`)  
- Visualize **feature importance** using Plotly bar charts  
- Color-coded prediction output for quick interpretation  

---

## 🧠 Tools & Technologies

| Category | Tools |
|-----------|--------|
| Machine Learning | scikit-learn (RandomForestClassifier / XGBoost) |
| Data Handling | Pandas, NumPy |
| Visualization | Plotly, Streamlit |
| Other | pickle, plotly.express, streamlit |

---

## ⚙️ How It Works

1. **Load Model & Scaler**  
   Loads the trained model (`best_model.pkl`) and `MinMaxScaler` (`scaler.pkl`) for input preprocessing.

2. **User Inputs**  
   Users input customer data through a Streamlit sidebar with number inputs and checkboxes.

3. **Preprocess Inputs**  
   - Converts boolean features to 0/1  
   - Scales numeric features with the loaded MinMaxScaler  

4. **Prediction**  
   - Model predicts churn probability and class (`Churned` / `Retain`)  
   - Output is color-coded: **red for Churn**, **green for Retain**  

5. **Feature Importance**  
   - Loads precomputed feature importance from `feature_importance.csv`  
   - Visualized using a horizontal Plotly bar chart  

---

## 📊 Sample Output

| Feature | Feature Importance Score |
|---------|-------------------------|
| Age | 0.12 |
| Balance | 0.15 |
| EstimatedSalary | 0.10 |
| Tenure | 0.08 |
| NumOfProducts | 0.07 |
| Geography_France | 0.05 |
| Geography_Spain | 0.04 |
| Gender_Female | 0.03 |

**Prediction Example:**  
- **Input:** Credit Score 650, Age 40, Balance 12000, etc.  
- **Output:** `Churned 🚨 (35.20% probability)` → Red  
- **Output:** `Retain ✅ (64.80% probability)` → Green  

---

