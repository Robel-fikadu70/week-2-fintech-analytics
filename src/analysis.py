import pandas as pd
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import os
import re

# SETUP: Download necessary NLTK datasets
# This usually runs once. If it fails, run 'python -m nltk.downloader all' in terminal.
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Downloading NLTK resources...")
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

INPUT_PATH = os.path.join("data", "cleaned_reviews.csv")
OUTPUT_PATH = os.path.join("data", "sentiment_analysis.csv")

def advanced_nlp_pipeline(text):
    """
    Clear NLP pipeline (Tokenization, Stop-words, Lemmatization)
    """
    text = str(text).lower()
    
    # 1. Tokenization (Splitting text into words)
    tokens = word_tokenize(text)
    
    # 2. Stop-word Removal (Removing common words like 'the', 'is', 'and')
    stop_words = set(stopwords.words('english'))
    # Adding domain-specific stopwords to surface real themes
    custom_stops = {'app', 'bank', 'mobile', 'money', 'ethiopia', 'phone', 'use'} 
    
    # 3. Lemmatization (Converting 'crashing' -> 'crash', 'transferred' -> 'transfer')
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for token in tokens:
        # Keep only alphanumeric tokens
        if token.isalnum() and token not in stop_words and token not in custom_stops:
            lemma = lemmatizer.lemmatize(token)
            clean_tokens.append(lemma)
            
    return " ".join(clean_tokens)

def extract_pain_points(df):
    """
    Keyword/n-gram extraction to surface concrete themes.
    Using Bi-grams (2 words) to catch phrases like "login error", "slow connection".
    """
    print("\n--- Top Themes & Pain Points (N-Grams) ---")
    
    # 'lemmatized_text' for better extraction
    vectorizer = TfidfVectorizer(ngram_range=(2, 2), max_features=5)
    
    for bank in df['bank_name'].unique():
        bank_df = df[df['bank_name'] == bank]
        
        # Only extract themes from Negative/Neutral reviews to find PAIN POINTS
        pain_df = bank_df[bank_df['sentiment_label'].isin(['Negative', 'Neutral'])]
        
        if not pain_df.empty:
            try:
                tfidf_matrix = vectorizer.fit_transform(pain_df['lemmatized_text'])
                keywords = vectorizer.get_feature_names_out()
                print(f"\n{bank} Pain Points:")
                for kw in keywords:
                    print(f"  - {kw}")
            except ValueError:
                print(f"\n{bank}: Not enough data for keyword extraction.")

def main():
    if not os.path.exists(INPUT_PATH):
        print("Data not found. Run scraper/cleaner first.")
        return

    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} reviews.")
    
    # --- TASK 2: SENTIMENT ANALYSIS ---
    print("Calculating Sentiment...")
    df['sentiment_score'] = df['review_text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    df['sentiment_label'] = df['sentiment_score'].apply(
        lambda x: 'Positive' if x > 0.1 else ('Negative' if x < -0.1 else 'Neutral')
    )

    # --- TASK 2: NLP PIPELINE ---
    print("Running NLP Pipeline (Tokenize -> Stopwords -> Lemmatize)...")
    df['lemmatized_text'] = df['review_text'].apply(advanced_nlp_pipeline)

    # --- TASK 2: THEME EXTRACTION ---
    extract_pain_points(df)

    # Save
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nAnalysis Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()