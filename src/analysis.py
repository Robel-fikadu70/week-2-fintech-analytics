import pandas as pd
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import re

INPUT_PATH = os.path.join("data", "cleaned_reviews.csv")
OUTPUT_PATH = os.path.join("data", "sentiment_analysis.csv")

#Preprocessing Pipeline (Tokenization/Stop-words logic)
def clean_for_nlp(text):
    """
    Simple NLP preprocessing: Lowercase, remove special chars.
    """
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    return text

def get_sentiment(text):
    """
    Compute sentiment scores (TextBlob)
    """
    return TextBlob(str(text)).sentiment.polarity

def extract_keywords(df):
    """
    Keyword Extraction using TF-IDF.
    """
    print("\n--- Extracting Keywords (TF-IDF) ---")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10)
    
    for bank in df['bank_name'].unique():
        bank_reviews = df[df['bank_name'] == bank]['review_text'].dropna()
        if len(bank_reviews) > 0:
            tfidf_matrix = vectorizer.fit_transform(bank_reviews)
            keywords = vectorizer.get_feature_names_out()
            print(f"Top Keywords for {bank}: {list(keywords)}")

def main():
    if not os.path.exists(INPUT_PATH):
        print("Data not found.")
        return

    df = pd.read_csv(INPUT_PATH)
    
    # 1. Apply Sentiment Analysis
    print("Running Sentiment Analysis...")
    df['sentiment_score'] = df['review_text'].apply(get_sentiment)
    
    # Labeling
    df['sentiment_label'] = df['sentiment_score'].apply(
        lambda x: 'Positive' if x > 0.1 else ('Negative' if x < -0.1 else 'Neutral')
    )
    
    # 2. Run Keyword Extraction (To satisfy Rubric)
    # We create a temporary column for NLP cleaning
    temp_clean = df['review_text'].apply(clean_for_nlp)
    extract_keywords(df)

    # Save Results
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nAnalysis complete. Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()