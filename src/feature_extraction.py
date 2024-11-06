# src/feature_extraction.py

import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer
import string

# Initialize stemmers for each language
ps_en = PorterStemmer()  # English stemmer
stemmer_de = SnowballStemmer('german')  # German stemmer
stemmer_fr = SnowballStemmer('french')  # French stemmer

# Hindi stopwords list
hindi_stopwords = {
    'और', 'का', 'कि', 'के', 'को', 'है', 'से', 'में', 'की', 'हैं', 'यह', 'पर', 'था', 'थे', 'लिए', 
    'अपने', 'इस', 'तो', 'भी', 'जा', 'हो', 'गया', 'कर', 'करने', 'रहा', 'तक', 'होता', 'नहीं', 'हूँ', 
    'इसी', 'जिस', 'जैसे', 'रहे', 'उस', 'यहां', 'किया', 'जब', 'तुम', 'अब', 'जो', 'सब', 'क्या'
}

def clean_english(text):
    text = re.sub(r'http\S+|www.\S+', '', text)  # Remove URLs
    text = re.sub(r'\W', ' ', text)  # Remove non-word characters
    text = text.lower().strip()
    return text

def clean_hindi(text):
    text = re.sub(r'[^\u0900-\u097F\s]', '', text)  # Keep only Devanagari script
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

def clean_german(text):
    text = re.sub(r'http\S+|www.\S+', '', text)  # Remove URLs
    text = re.sub(r'ä', 'ae', text)
    text = re.sub(r'ö', 'oe', text)
    text = re.sub(r'ü', 'ue', text)
    text = re.sub(r'ß', 'ss', text)
    text = re.sub(r'\W', ' ', text)  # Remove non-word characters
    text = text.lower().strip()
    return text

def clean_french(text):
    text = re.sub(r'http\S+|www.\S+', '', text)  # Remove URLs
    text = re.sub(r'é|è|ê', 'e', text)
    text = re.sub(r'à|â', 'a', text)
    text = re.sub(r'î', 'i', text)
    text = re.sub(r'ô', 'o', text)
    text = re.sub(r'ç', 'c', text)
    text = re.sub(r'\W', ' ', text)  # Remove non-word characters
    text = text.lower().strip()
    return text

def clean_text_by_language(text, language):
    """
    Clean text based on the specified language.
    """
    if language == 'en':
        return clean_english(text)
    elif language == 'hi':
        return clean_hindi(text)
    elif language == 'de':
        return clean_german(text)
    elif language == 'fr':
        return clean_french(text)
    else:
        return text  # For unknown languages, return the text as-is

def transform_text_en(text):
    text = text.lower()
    text = word_tokenize(text, preserve_line=True)  # Ensure line preservation to avoid sentence tokenization
    text = [word for word in text if word.isalnum()]
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    text = [ps_en.stem(word) for word in text]
    return " ".join(text)

def preprocess_hindi_text(text):
    text = text.lower()
    words = word_tokenize(text, preserve_line=True)  # Ensure line preservation
    words = [word for word in words if re.match(r'[\u0900-\u097F]+$', word)]
    words = [word for word in words if word not in hindi_stopwords]
    words = [re.sub(r'(ों|ें|ो|े|ा|ि|ी|ु|ू|ं|ः)$', '', word) for word in words]
    return ' '.join(words)

def transform_text_de(text):
    text = text.lower()
    text = word_tokenize(text, preserve_line=True)
    text = [word for word in text if word.isalnum()]
    text = [word for word in text if word not in stopwords.words('german') and word not in string.punctuation]
    text = [stemmer_de.stem(word) for word in text]
    return " ".join(text)

def transform_text_fr(text):
    text = text.lower()
    text = word_tokenize(text, preserve_line=True)
    text = [word for word in text if word.isalnum()]
    text = [word for word in text if word not in stopwords.words('french') and word not in string.punctuation]
    text = [stemmer_fr.stem(word) for word in text]
    return " ".join(text)

def transform_text(text, language):
    if language == 'en':
        return transform_text_en(text)
    elif language == 'hi':
        return preprocess_hindi_text(text)
    elif language == 'de':
        return transform_text_de(text)
    elif language == 'fr':
        return transform_text_fr(text)
    else:
        return text  