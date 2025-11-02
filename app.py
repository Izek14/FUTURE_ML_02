from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import pickle
import pandas as pd
import plotly.express as px

# Set Streamlit Layout to wide
st.set_page_config(layout="wide")

# Load the trained model
with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the MinMaxScaler
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Display names -> actual model feature names
feature_label_map = {
    "Credit Score": "CreditScore",
    "Customer Age": "Age",
    "Tenure": "Tenure",
    "Account Balance": "Balance",
    "Number of Products": "NumOfProducts",
    "Estimated Salary": "EstimatedSalary",
    "Lives in France": "Geography_France",
    "Lives in Germany": "Geography_Germany",
    "Lives in Spain": "Geography_Spain",
    "Female": "Gender_Female",
    "Male": "Gender_Male",
    "No Credit Card": "HasCrCard_0",
    "Has Credit Card": "HasCrCard_1",
    "Not Active Member": "IsActiveMember_0",
    "Active Member": "IsActiveMember_1"
}

# Define the input features for the model
feature_names = list(feature_label_map.values())

# Columns requiring scaling
scale_vars = ["CreditScore", "EstimatedSalary", "Tenure", "Balance", "Age", "NumOfProducts"]

# Updated default values
default_values = [
    600, 30, 2, 8000, 2, 60000,
    True, False, False, True, False, False, True, False, True
]

# Sidebar setup
st.sidebar.header("User Inputs")

# Collect user inputs
user_inputs_display = {}
for (display_name, feature_name), default in zip(feature_label_map.items(), default_values):
    if feature_name in scale_vars:
        user_inputs_display[display_name] = st.sidebar.number_input(
            display_name, value=default, step=1 if isinstance(default, int) else 0.01
        )
    elif isinstance(default, bool):
        user_inputs_display[display_name] = st.sidebar.checkbox(display_name, value=default)
    else:
        user_inputs_display[display_name] = st.sidebar.number_input(display_name, value=default, step=1)

# Convert display inputs to actual model feature keys
user_inputs = {feature_label_map[k]: v for k, v in user_inputs_display.items()}

# Convert inputs to a DataFrame
input_data = pd.DataFrame([user_inputs])

# Apply MinMaxScaler to the required columns
input_data_scaled = input_data.copy()
input_data_scaled[scale_vars]= scaler.transform(input_data[scale_vars])

# App Header
st.title("Customer Churn Prediction")

# Page Layout
left_col, right_col = st.columns(2)

# Left Page: Feature Importance
with left_col:
    st.header("Feature Importance")
    # Load feature importance data from the Excel file
    feature_importance_df = pd.read_csv("feature_importance.csv", usecols=["Feature", "Feature Importance Score"])
    # Plot the feature importance bar chart
    fig = px.bar(
        feature_importance_df.sort_values(by="Feature Importance Score", ascending=True),
        x="Feature Importance Score",
        y="Feature",
        orientation="h",
        title="Feature Importance",
        labels={"Feature Importance Score": "Importance", "Feature": "Features"},
        width=400,
        height=500
    )
    st.plotly_chart(fig)

# Right Page: Prediction
with right_col:
    st.header("Prediction")
    if st.button("Predict"):
        # Get the predicted probabilities and label
        probabilities = model.predict_proba(input_data_scaled)[0]
        prediction = model.predict(input_data_scaled)[0]
        # Map prediction to label
        prediction_label = "Churned" if prediction == 1 else "Retain"
        
        # Display results
        st.subheader(f"Predicted Value: {prediction_label}")
        st.error(f"Predicted Probability: {probabilities[1] :.2%} (Churn)")
        st.success(f"Predicted Probability: {probabilities[0] :.2%} (Retain)")
        # Display a clear output for the prediction
        if prediction == 1:
            st.markdown(f"<h3>Output: <span style='color:red'>{prediction_label} 🚨 ({probabilities[1]:.2%})</span></h3>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h3>Output: <span style='color:green'->{prediction_label} ✅ ({probabilities[0]:.2%})</span></h3>", unsafe_allow_html=True)

# Streamlit run app.py