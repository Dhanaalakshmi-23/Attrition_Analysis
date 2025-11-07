# app.py
# IBM HR Analytics: Employee Attrition EDA + Prediction Dashboard
# ---------------------------------------------------------------
# Author: [Your Name]
# Date: [Today's Date]
# Description: Streamlit dashboard for EDA and predictive modeling on Employee Attrition dataset

# -------------------------------
# Imports
# -------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# -------------------------------
# Streamlit App Configuration
# -------------------------------
st.set_page_config(page_title="HR Attrition Analysis", layout="wide")

# -------------------------------
# Load Data
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
    return df

df = load_data()

# -------------------------------
# Sidebar Filters
# -------------------------------
st.sidebar.header("Filter Options")
departments = df["Department"].unique()
genders = df["Gender"].unique()
roles = df["JobRole"].unique()

selected_dept = st.sidebar.multiselect("Select Department(s)", departments, default=departments)
selected_gender = st.sidebar.multiselect("Select Gender(s)", genders, default=genders)
selected_role = st.sidebar.multiselect("Select Job Role(s)", roles, default=roles)

filtered_df = df[
    (df["Department"].isin(selected_dept)) &
    (df["Gender"].isin(selected_gender)) &
    (df["JobRole"].isin(selected_role))
]

# -------------------------------
# Overview Section
# -------------------------------
st.title("📊 HR Attrition EDA and Prediction Dashboard")

st.subheader("Data Overview")

total_employees = len(filtered_df)
attritions = filtered_df[filtered_df["Attrition"] == "Yes"].shape[0]
attrition_rate = attritions / total_employees * 100
avg_age = filtered_df["Age"].mean()
avg_income = filtered_df["MonthlyIncome"].mean()
avg_satisfaction = filtered_df["JobSatisfaction"].mean()

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Total Employees", f"{total_employees}")
col2.metric("Attritions", f"{attritions}")
col3.metric("Attrition Rate (%)", f"{attrition_rate:.2f}")
col4.metric("Avg Age", f"{avg_age:.1f}")
col5.metric("Avg Salary", f"${avg_income:,.0f}")
col6.metric("Avg Job Satisfaction", f"{avg_satisfaction:.2f}")

# -------------------------------
# Demographic Analysis
# -------------------------------
st.markdown("### Demographic Analysis")

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(px.histogram(filtered_df, x="Age", color="Attrition", nbins=20,
                                 title="Attrition by Age"), use_container_width=True)
with col2:
    st.plotly_chart(px.pie(filtered_df, names="Gender", title="Gender Distribution"), use_container_width=True)

col3, col4 = st.columns(2)
with col3:
    edu_chart = filtered_df.groupby(["EducationField", "Attrition"]).size().reset_index(name="Count")
    st.plotly_chart(px.bar(edu_chart, x="EducationField", y="Count", color="Attrition",
                           title="Attrition by Education Field"), use_container_width=True)
with col4:
    marital_chart = filtered_df.groupby(["MaritalStatus", "Attrition"]).size().reset_index(name="Count")
    st.plotly_chart(px.bar(marital_chart, x="MaritalStatus", y="Count", color="Attrition",
                           title="Attrition by Marital Status"), use_container_width=True)

# -------------------------------
# Work-Related Analysis
# -------------------------------
st.markdown("### Work-Related Analysis")

col1, col2 = st.columns(2)
with col1:
    dep_chart = filtered_df.groupby(["Department", "Attrition"]).size().reset_index(name="Count")
    st.plotly_chart(px.bar(dep_chart, x="Department", y="Count", color="Attrition",
                           title="Attrition by Department"), use_container_width=True)
with col2:
    travel_chart = filtered_df.groupby(["BusinessTravel", "Attrition"]).size().reset_index(name="Count")
    st.plotly_chart(px.bar(travel_chart, x="BusinessTravel", y="Count", color="Attrition",
                           title="Attrition by Business Travel"), use_container_width=True)

st.plotly_chart(px.scatter(filtered_df, x="YearsAtCompany", y="MonthlyIncome", color="Attrition",
                           title="Monthly Income vs Years at Company"), use_container_width=True)

# -------------------------------
# Performance and Satisfaction
# -------------------------------
st.markdown("### Performance and Satisfaction")

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(px.box(filtered_df, x="Attrition", y="JobSatisfaction", color="Attrition",
                           title="Job Satisfaction vs Attrition"), use_container_width=True)
with col2:
    st.plotly_chart(px.box(filtered_df, x="Attrition", y="PerformanceRating", color="Attrition",
                           title="Performance Rating vs Attrition"), use_container_width=True)

# -------------------------------
# Correlation Insights
# -------------------------------
st.markdown("### Correlation Insights")

num_cols = filtered_df.select_dtypes(include=np.number)
fig_corr, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(num_cols.corr(), cmap="coolwarm")
st.pyplot(fig_corr)

# -------------------------------
# Machine Learning: Attrition Prediction
# -------------------------------
st.markdown("### Predict Employee Attrition")

ml_df = df.copy()
le = LabelEncoder()
for col in ml_df.select_dtypes(include='object').columns:
    ml_df[col] = le.fit_transform(ml_df[col])

X = ml_df.drop(["Attrition"], axis=1)
y = ml_df["Attrition"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)

st.write(f"Model Accuracy: {acc * 100:.2f}%")
st.text("Confusion Matrix:")
st.write(confusion_matrix(y_test, y_pred))
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# -------------------------------
# Insights Summary
# -------------------------------
st.markdown("### Insights Summary")

st.write("""
- Higher attrition among employees with frequent business travel.
- Employees with low job satisfaction tend to leave more often.
- Younger employees and those with fewer years at the company are more likely to quit.
- Departments with lower average pay, particularly Sales, see higher attrition.
- Married employees show lower turnover rates compared to single employees.
""")

# -------------------------------
# Optional Export
# -------------------------------
if st.button("Export Insights to Excel"):
    filtered_df.to_excel("HR_Attrition_Insights.xlsx", index=False)
    st.success("Exported as HR_Attrition_Insights.xlsx")
