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

# Function to preprocess and train the model
def train_model(df):
    # Preprocess the data
    X = df['externalStatus']
    y = df['internalStatus']
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    # Tokenize and encode the input data
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(X)
    X = X.toarray()

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Determine input shape
    input_shape = X_train.shape[1]

    # Build the model architecture
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(len(encoder.classes_), activation='softmax')
    ])

    # Compile the model
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, verbose=0)

    # Evaluate the model
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    accuracy = accuracy_score(y_test, y_pred_classes)
    precision = precision_score(y_test, y_pred_classes, average='weighted')
    recall = recall_score(y_test, y_pred_classes, average='weighted')
    f1 = f1_score(y_test, y_pred_classes, average='weighted')

    return model, accuracy, precision, recall, f1

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
