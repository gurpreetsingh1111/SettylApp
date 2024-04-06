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

    # Print shapes for debugging
    print(X_train.shape, y_train.shape)

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

    return model, accuracy, precision, recall, f1, y_test, y_pred_classes, encoder

# Streamlit app
def main():
    st.title("🌟🚀Welcome to My SettyAI App🚀🌟")

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
        model, accuracy, precision, recall, f1, y_test, y_pred_classes, encoder = train_model(df)

        # Display evaluation metrics
        st.write("Model Evaluation Metrics:")
        st.write(f"- Accuracy: {accuracy}")
        st.write(f"- Precision: {precision}")
        st.write(f"- Recall: {recall}")
        st.write(f"- F1-score: {f1}")

        # Button to display accuracy
        if st.button("Show Accuracy"):
            st.write("Accuracy:", accuracy)

        # Calculate evaluation metrics
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
        values = [accuracy, precision, recall, f1]

        # Create bar plot
        st.write("Bar plot of Model Evaluation Metrics:")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(metrics, values, color=['blue', 'green', 'orange', 'red'])
        ax.set_title('Model Evaluation Metrics')
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_ylim(0.9, 1)  # Set y-axis limit to better visualize differences
        st.pyplot(fig)

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

        # Create DataFrame for predicted and actual data
        df_predicted_actual = pd.DataFrame({
            'Actual': encoder.inverse_transform(y_test),
            'Predicted': encoder.inverse_transform(y_pred_classes)
        })

        # Provide download button for predicted and actual data
        st.write("Download predicted and actual data:")
        csv = df_predicted_actual.to_csv(index=False)
        st.download_button(label="Download Data", data=csv, file_name='predicted_actual_data.csv', mime='text/csv')

if __name__ == "__main__":
    main()
