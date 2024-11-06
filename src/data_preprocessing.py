# src/data_preprocessing.py

import pandas as pd

def load_and_process_data(file_path):
    try:
        # Attempt to load with UTF-8 encoding
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        # Fallback to ISO-8859-1 if UTF-8 fails
        df = pd.read_csv(file_path, encoding='ISO-8859-1')
    
    # Process data if needed
    df = df.melt(id_vars=['labels'], value_vars=['text', 'text_hi', 'text_de', 'text_fr'],
                 var_name='language_column', value_name='message')
    language_map = {'text': 'en', 'text_hi': 'hi', 'text_de': 'de', 'text_fr': 'fr'}
    df['language'] = df['language_column'].map(language_map)
    df = df.drop(columns=['language_column'])
    
    return df
