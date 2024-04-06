import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = "https://github.com/gurpreetsingh1111/SettylApp/blob/main/trained_model.pkl"

# Define function to predict internal status
def predict_internal_status(external_status):
    try:
        predicted_label = model.predict([external_status])[0]
        return predicted_label
    except Exception as e:
        return str(e)

# Function to read CSV dataset
def read_csv_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        return None

# Streamlit app
def main():
    # Title and description
    st.title("Internal Status Prediction")
    st.write("Enter an external status description to predict the internal status.")

    # Input field for external status description
    external_status = st.text_input("External Status Description:", "")

    # Button to trigger prediction
    if st.button("Predict"):
        if external_status:
            # Call prediction function
            predicted_internal_status = predict_internal_status(external_status)
            st.success(f"Predicted Internal Status: {predicted_internal_status}")
        else:
            st.warning("Please enter an external status description.")

    # Sidebar - Collects user input features into dataframe
    with st.sidebar.header('1. Upload your CSV data'):
        uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
        st.sidebar.markdown("""
    [Example CSV input file](https://github.com/gurpreetsingh1111/SettylApp/blob/main/finaldata.csv)
    """)

        # Read uploaded CSV file
        if uploaded_file is not None:
            df = read_csv_dataset(uploaded_file)
            if df is not None:
                st.sidebar.write("Dataset loaded successfully:")
                st.sidebar.write(df)
            else:
                st.sidebar.error("Failed to load dataset. Please check the file format.")

if __name__ == "__main__":
    main()
