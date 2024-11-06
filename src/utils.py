# src/utils.py

import nltk
import pandas as pd

def download_stopwords():
    nltk.download('stopwords')

def append_to_dataset(file_path, new_data):
    """Append new data to the dataset and save it."""
    df_new = pd.DataFrame([new_data])
    try:
        # Append to existing file without headers
        df_new.to_csv(file_path, mode='a', header=False, index=False)
        print("New data added successfully.")
    except Exception as e:
        print(f"Error saving new data: {e}")