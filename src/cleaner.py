import pandas as pd
import os

RAW_PATH = os.path.join("data", "raw_reviews.csv")
CLEAN_PATH = os.path.join("data", "cleaned_reviews.csv")

def preprocess_text(df):
    """
    Rubric: Duplicates removed, missing data handled.
    """
    initial_count = len(df)
    
    # 1. Rename columns to match PDF requirements
    # Scraper gives: content, score, at -> PDF wants: review, rating, date
    df = df.rename(columns={
        'content': 'review_text',
        'score': 'rating',
        'at': 'review_date'
    })
    
    # 2. Drop Missing Reviews (<5% missing data KPI)
    df = df.dropna(subset=['review_text'])
    
    # 3. Drop Duplicates
    df = df.drop_duplicates(subset=['review_text', 'bank_name'])
    
    # 4. Normalize Date (YYYY-MM-DD)
    df['review_date'] = pd.to_datetime(df['review_date']).dt.strftime('%Y-%m-%d')
    
    # Select only required columns
    required_cols = ['review_text', 'rating', 'review_date', 'bank_name', 'source']
    df = df[required_cols]
    
    print(f"Cleaning Report: {initial_count} -> {len(df)} reviews.")
    return df

def main():
    if not os.path.exists(RAW_PATH):
        print("Run scraper.py first!")
        return

    df = pd.read_csv(RAW_PATH)
    clean_df = preprocess_text(df)
    
    clean_df.to_csv(CLEAN_PATH, index=False)
    print(f"Cleaned data saved to {CLEAN_PATH}")

if __name__ == "__main__":
    main()