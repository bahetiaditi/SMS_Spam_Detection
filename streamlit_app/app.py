# app.py

import os
import sys
import nltk
import streamlit as st
import pandas as pd

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import functions and modules
from src.data_preprocessing import load_and_process_data
from src.utils import download_stopwords, append_to_dataset
from src.feature_extraction import clean_text_by_language, transform_text
from src.train_model import train_models_for_all_languages

# Clear cached data
st.cache_data.clear()

# Set the file path for the dataset
file_path = '/Users/aditibaheti/Machine_learning_projects/SMS_Spam_Detection/SMS_Spam_Detection/Data/Raw/data-en-hi-de-fr.csv'  # Update to the actual path of your dataset

# Load and process data
df_melted = load_and_process_data(file_path)
df_melted['labels'] = df_melted['labels'].map({'ham': 0, 'spam': 1})

# Apply cleaning and transformation
df_melted['cleaned_message'] = df_melted.apply(
    lambda row: clean_text_by_language(row['message'], row['language']), axis=1
)
df_melted['transformed_text'] = df_melted.apply(
    lambda row: transform_text(row['cleaned_message'], row['language']), axis=1
)
df_melted['transformed_text'] = df_melted['transformed_text'].apply(lambda x: "empty_message" if x.strip() == "" else x)

language_models = train_models_for_all_languages(df_melted)

# Streamlit UI elements
st.title("Multilingual SMS Spam Detector")

# Language selection dropdown
language_choice = st.selectbox("Select a Language", options=['en', 'hi', 'de', 'fr'], format_func=lambda x: x.upper())

# Text input for message classification
user_message = st.text_area("Enter the message to classify")

# True label input (radio button)
true_label = st.radio("What is the true label for this message?", options=['ham', 'spam'], index=0)

# Button to classify the message
if st.button("Classify Message"):
    def classify_and_save_message(message, lang_choice, true_lbl):
        """
        Classify the message and append it to the dataset with the true label.
        """
        # Get the relevant model for the chosen language
        model_data = language_models.get(lang_choice)
        if model_data:
            # Preprocess and transform the input message
            cleaned_message = clean_text_by_language(message, lang_choice)
            transformed_message = transform_text(cleaned_message, lang_choice)
            X_input = model_data['tfidf'].transform([transformed_message]).toarray()
            X_input = model_data['scaler'].transform(X_input)

            # Predict the message classification
            prediction = model_data['model'].predict(X_input)[0]
            label = "Spam" if prediction == 1 else "Ham"
            st.write(f"### The message is classified as: {label}")

            # Map the true label to numeric
            numeric_label = 1 if true_lbl == 'spam' else 0

            # Create new data entry with message and label information
            new_data = {
                'language': lang_choice,
                'message': message,
                'cleaned_message': cleaned_message,
                'transformed_text': transformed_message,
                'labels': numeric_label
            }
            append_to_dataset(file_path, new_data)
            st.write("New data added successfully.")
        else:
            st.write("Model not available for the selected language.")

    # Call the classification function
    classify_and_save_message(user_message, language_choice, true_label)
