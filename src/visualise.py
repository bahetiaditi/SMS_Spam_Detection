# src/visualise.py

from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
from matplotlib.font_manager import FontProperties
import pandas as pd

def create_wordcloud(text, title, font_path=None):
    if not text or text.strip() == "empty_message":
        print(f"Skipping word cloud generation for '{title}' as it has no valid content.")
        return
    wc = WordCloud(width=1000, height=1000, min_font_size=10, background_color='black', font_path=font_path)
    plt.imshow(wc.generate(text))
    plt.axis("off")
    plt.title(title)
    plt.show()

def plot_wordclouds(df, font_path):
    for language in df['language'].unique():
        # Filter for spam and ham for each language and join texts
        spam_text = " ".join(df[(df['labels'] == 1) & (df['language'] == language)]['transformed_text'].dropna())
        ham_text = " ".join(df[(df['labels'] == 0) & (df['language'] == language)]['transformed_text'].dropna())

        # Display word clouds only if there is content
        plt.figure(figsize=(10, 5))
        if spam_text.strip() and spam_text != "empty_message":
            plt.subplot(1, 2, 1)
            create_wordcloud(spam_text, f"{language.upper()} Spam Word Cloud", font_path if language == 'hi' else None)
        else:
            print(f"No spam messages for {language.upper()} to generate a word cloud.")

        if ham_text.strip() and ham_text != "empty_message":
            plt.subplot(1, 2, 2)
            create_wordcloud(ham_text, f"{language.upper()} Ham Word Cloud", font_path if language == 'hi' else None)
        else:
            print(f"No ham messages for {language.upper()} to generate a word cloud.")