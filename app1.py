import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pk
import joblib


model = joblib.load("iris_model.pkl")


st.title("Iris Species Prediction App")
st.write("This app predicts the species of an Iris flower based on its features.")
st.image("https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg", width=300)


sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.0)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.0)

features = np.array([[float(sepal_length), float(sepal_width), float(petal_length), float(petal_width)]])



if st.button("Predict Species"):
    prediction = model.predict(features)
    class_names = ['Setosa', 'Versicolor', 'Virginica']
    st.success(f"The predicted species is: {class_names[prediction[0]]}")