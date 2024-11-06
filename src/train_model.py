# src/train_model.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix

# Dictionary to hold trained models for each language
language_models = {}
results = {}

def train_model_for_language(language, df_language):
    # Ensure no NaN values in labels
    df_language = df_language.dropna(subset=['labels'])
    
    # Feature extraction
    tfidf = TfidfVectorizer(max_features=3000)
    X = tfidf.fit_transform(df_language['transformed_text']).toarray()
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    y = df_language['labels'].values
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=2
    )
    
    # Initialize and train model
    mnb = MultinomialNB()
    mnb.fit(X_train, y_train)
    
    # Evaluate model
    accuracy = accuracy_score(y_test, mnb.predict(X_test))
    precision = precision_score(y_test, mnb.predict(X_test))
    conf_matrix = confusion_matrix(y_test, mnb.predict(X_test))
    
    # Return the model, tfidf, scaler, and evaluation metrics
    return {
        'model': mnb,
        'tfidf': tfidf,
        'scaler': scaler,
        'accuracy': accuracy,
        'precision': precision,
        'confusion_matrix': conf_matrix
    }

def train_models_for_all_languages(df):
    language_models = {}
    for language in df['language'].unique():
        df_language = df[df['language'] == language]
        language_models[language] = train_model_for_language(language, df_language)
    return language_models