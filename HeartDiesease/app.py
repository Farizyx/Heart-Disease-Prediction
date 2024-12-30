import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score

# Load dataset (replace this path with your actual dataset path)
@st.cache
def load_data():
    # Example path to the dataset (replace with your actual dataset)
    data = pd.read_csv('heart_disease_data.csv')  # Make sure to update the path
    return data

# Preprocess the dataset (balance it using SMOTE)
@st.cache
def preprocess_data(data):
    # Features (X) and target (y)
    X = data.drop(columns=['heart_disease_present'])
    y = data['heart_disease_present']
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Balance the dataset using SMOTE
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    return X_train_balanced, X_test, y_train_balanced, y_test, scaler

# Train KNN model
@st.cache
def train_model(X_train, y_train):
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    return model

# Evaluate model
@st.cache
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Load the data and preprocess
data = load_data()
X_train_balanced, X_test, y_train_balanced, y_test, scaler = preprocess_data(data)

# Train the model on the balanced dataset
model = train_model(X_train_balanced, y_train_balanced)

# Evaluate the model's accuracy
accuracy = evaluate_model(model, X_test, y_test)

# Streamlit Sidebar Header
st.sidebar.header('User Input Features')

# Function to get user input
def user_input_features():
    # Get inputs from the user
    age = st.sidebar.number_input('Enter Your Age:', min_value=1, max_value=120, value=25)
    sex = st.sidebar.selectbox('Sex', (0, 1))  # 0 for Female, 1 for Male
    chest_pain_type = st.sidebar.selectbox('Chest Pain Type', (0, 1, 2, 3))  # Example of chest pain types (replace with actual)
    resting_blood_pressure = st.sidebar.number_input('Resting Blood Pressure:', min_value=50, max_value=200, value=120)
    serum_cholesterol = st.sidebar.number_input('Serum Cholesterol:', min_value=100, max_value=600, value=200)
    fasting_blood_sugar = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl', (0, 1))  # 0 = No, 1 = Yes
    resting_ecg = st.sidebar.selectbox('Resting Electrocardiographic Results', (0, 1, 2))  # Example ECG results
    max_heart_rate = st.sidebar.number_input('Maximum Heart Rate:', min_value=60, max_value=200, value=150)
    exercise_induced_angina = st.sidebar.selectbox('Exercise Induced Angina', (0, 1))  # 0 = No, 1 = Yes
    oldpeak = st.sidebar.number_input('Oldpeak (Depression):', min_value=0.0, max_value=6.0, value=1.0)
    slope = st.sidebar.selectbox('Slope of Peak Exercise ST Segment', (0, 1, 2))  # Example slopes
    ca = st.sidebar.selectbox('Number of Major Vessels Colored by Fluoroscopy', (0, 1, 2, 3))
    thalassemia = st.sidebar.selectbox('Thalassemia', (1, 2, 3))  # Example thalassemia types

    # Return the values as a DataFrame (for easier processing)
    user_data = pd.DataFrame(
        [[age, sex, chest_pain_type, resting_blood_pressure, serum_cholesterol, fasting_blood_sugar,
          resting_ecg, max_heart_rate, exercise_induced_angina, oldpeak, slope, ca, thalassemia]],
        columns=['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'serum_cholesterol', 
                 'fasting_blood_sugar', 'resting_ecg', 'max_heart_rate', 'exercise_induced_angina', 
                 'oldpeak', 'slope', 'ca', 'thalassemia'])
    
    return user_data

# Get user input
user_input = user_input_features()

# Display user input
st.subheader('User Input Features')
st.write(user_input)

# Standardize the user input
user_input_scaled = scaler.transform(user_input)

# Make prediction using the trained model
prediction = model.predict(user_input_scaled)

# Show prediction result
st.subheader('Prediction Result')
if prediction[0] == 0:
    st.write("No heart disease detected.")
else:
    st.write("Heart disease detected.")

# Show model accuracy
st.subheader('Model Accuracy')
st.write(f'Model Accuracy: {accuracy * 100:.2f}%')
