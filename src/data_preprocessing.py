# src/data_processing.py

import pandas as pd

def load_and_process_data(file_path):
    """
    Load the dataset, reshape using melt, and map language columns to language codes.
    """
    df = pd.read_csv(file_path)
    df_melted = df.melt(id_vars=['labels'], value_vars=['text', 'text_hi', 'text_de', 'text_fr'],
                        var_name='language_column', value_name='message')

    # Map language columns to language codes
    language_map = {'text': 'en', 'text_hi': 'hi', 'text_de': 'de', 'text_fr': 'fr'}
    df_melted['language'] = df_melted['language_column'].map(language_map)

    # Drop the original language column indicator
    df_melted = df_melted.drop(columns=['language_column'])
    
    return df_melted
