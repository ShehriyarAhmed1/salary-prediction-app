import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, classification_report, accuracy_score
from imblearn.over_sampling import RandomOverSampler
import warnings
warnings.filterwarnings("ignore")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("ds_salaries.csv")
    return df

df = load_data()

st.title("💼 Data Science Salary Prediction App")

# Sidebar for task selection
task = st.sidebar.radio("Choose Task", ["Regression", "Classification", "EDA"])

if task == "EDA":
    st.subheader("🔍 Exploratory Data Analysis")

    st.write("### Dataset Overview")
    st.dataframe(df.head())

    st.write("### Summary Statistics")
    st.write(df.describe())

    st.write("### Correlation Heatmap")
    plt.figure(figsize=(10, 5))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
    st.pyplot(plt)

    st.write("### Categorical Features")
    cat_cols = df.select_dtypes(include='object').columns
    st.write(df[cat_cols].nunique())

    st.write("### Salary Distribution")
    plt.figure(figsize=(8, 4))
    sns.histplot(df['salary_in_usd'], bins=30, kde=True)
    st.pyplot(plt)

else:
    st.subheader(f"🚀 {task} Model")

    # Preprocessing
    df_model = df.copy()
    df_model.drop(columns=["salary", "salary_currency", "employee_residence", "company_location"], inplace=True)

    # Label Encoding for categorical features
    le = LabelEncoder()
    for col in df_model.select_dtypes(include='object').columns:
        df_model[col] = le.fit_transform(df_model[col])

    if task == "Regression":
        X = df_model.drop("salary_in_usd", axis=1)
        y = df_model["salary_in_usd"]

        model_choice = st.selectbox("Select Regression Model", ["Linear Regression", "Random Forest"])
        test_size = st.slider("Test Size (%)", 10, 50, 20)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

        if model_choice == "Linear Regression":
            model = LinearRegression()
        else:
            model = RandomForestRegressor()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.write("### Model Performance")
        st.write(f"R² Score: {r2_score(y_test, y_pred):.2f}")
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        st.write(f"RMSE: {rmse:.2f}")

        st.write("### Feature Importance")
        try:
            feat_importance = model.feature_importances_
            feat_df = pd.DataFrame({'Feature': X.columns, 'Importance': feat_importance})
            feat_df = feat_df.sort_values(by="Importance", ascending=False)

            plt.figure(figsize=(10, 4))
            sns.barplot(x="Importance", y="Feature", data=feat_df)
            st.pyplot(plt)
        except:
            st.info("Feature importance not available for this model.")

    else:
        # Classification Task (High vs Low salary classification)
        salary_threshold = st.slider("Set Salary Threshold", 50000, 300000, 100000)
        df_model["salary_class"] = df_model["salary_in_usd"].apply(lambda x: 1 if x >= salary_threshold else 0)
        X = df_model.drop(["salary_in_usd", "salary_class"], axis=1)
        y = df_model["salary_class"]

        model_choice = st.selectbox("Select Classification Model", ["Logistic Regression", "Random Forest"])
        balance_data = st.checkbox("Apply Random Oversampling to balance classes", value=True)
        test_size = st.slider("Test Size (%)", 10, 50, 20)

        if balance_data:
            ros = RandomOverSampler(random_state=42)
            X, y = ros.fit_resample(X, y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        if model_choice == "Logistic Regression":
            model = LogisticRegression()
        else:
            model = RandomForestClassifier()

        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        st.write("### Classification Report")
        st.text(classification_report(y_test, y_pred))

        st.write("### Accuracy")
        st.write(f"{accuracy_score(y_test, y_pred) * 100:.2f}%")

        st.write("### Feature Importance")
        try:
            feat_importance = model.feature_importances_
            feat_df = pd.DataFrame({'Feature': X.columns, 'Importance': feat_importance})
            feat_df = feat_df.sort_values(by="Importance", ascending=False)

            plt.figure(figsize=(10, 4))
            sns.barplot(x="Importance", y="Feature", data=feat_df)
            st.pyplot(plt)
        except:
            st.info("Feature importance not available for this model.")

    st.subheader("🧠 SHAP Explainability")
    st.info("SHAP explainability is temporarily disabled due to compatibility issues. You can enable it later if needed.")