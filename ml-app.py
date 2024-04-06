import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Function to preprocess text
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    # Join the lemmatized tokens back into a single string
    preprocessed_text = ' '.join(lemmatized_tokens)
    return preprocessed_text

# Streamlit app
def main():
    st.title("Internal Status Prediction")

    # Upload CSV file
    st.sidebar.header('1. Upload your CSV data')
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("Dataset loaded successfully.")

        # Preprocess the data
        df['externalStatus_processed'] = df['externalStatus'].apply(preprocess_text)
        df['internalStatus_processed'] = df['internalStatus'].apply(preprocess_text)

        # Train the model and get evaluation metrics
        model, accuracy, precision, recall, f1 = train_model(df)

        # Display evaluation metrics
        st.write("Model Evaluation Metrics:")
        st.write(f"- Accuracy: {accuracy}")
        st.write(f"- Precision: {precision}")
        st.write(f"- Recall: {recall}")
        st.write(f"- F1-score: {f1}")

        # User input for prediction
        st.write("Enter an external status description to predict the internal status.")
        external_status = st.text_input("External Status Description:")

        if st.button("Predict"):
            if external_status:
                # Preprocess input for prediction
                vectorizer = CountVectorizer()
                external_status_encoded = vectorizer.fit_transform([external_status])
                external_status_encoded = external_status_encoded.toarray()

                # Make prediction
                predicted_label = model.predict(external_status_encoded)[0]
                st.success(f"Predicted Internal Status: {predicted_label}")
            else:
                st.warning("Please enter an external status description.")

        # Plot actual vs predicted data
        st.write("Plot of Actual vs Predicted Data:")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(y_test, label='Actual', color='blue')
        ax.plot(y_pred_classes, label='Predicted', color='red', linestyle='dashed')
        ax.set_title('Actual vs Predicted Data')
        ax.set_xlabel('Data Point')
        ax.set_ylabel('Class')
        ax.legend()
        st.pyplot(fig)

if __name__ == "__main__":
    main()
