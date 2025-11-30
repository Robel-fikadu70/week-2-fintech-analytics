import pandas as pd
from google_play_scraper import Sort, reviews
import os

# CONSTANTS
APPS = {
    'CBE': 'com.combanketh.mobilebanking',
    'BOA': 'com.bankofabyssinia.boamobile.retail', 
    'Dashen': 'com.dashen.dashensuperapp'
}

RAW_DATA_PATH = os.path.join("data", "raw_reviews.csv")

def scrape_bank_reviews(bank_name, app_id, target_count=450):
    """
    Scrapes reviews for a specific bank. 
    """
    print(f"--- Scraping {bank_name} ---")
    
    # 1. Try Scraping (US Store first)
    try:
        result, _ = reviews(
            app_id,
            lang='en', 
            country='us', 
            sort=Sort.NEWEST, 
            count=target_count
        )
    except Exception as e:
        result = []
        print(f"  [US Store] Failed: {e}")

    # 2. Fallback: Try Ethiopia Store (country='et') if US returned 0
    if len(result) == 0:
        print(f"  [Info] US store returned 0 reviews. Retrying with country='et'...")
        try:
            result, _ = reviews(
                app_id,
                lang='en', 
                country='et', # Switch to Ethiopia store
                sort=Sort.NEWEST, 
                count=target_count
            )
        except Exception as e:
            print(f"  [ET Store] Failed: {e}")

    # Add identifiers
    for r in result:
        r['bank_name'] = bank_name
        r['source'] = 'Google Play'
        
    if len(result) > 0:
        print(f"  Success: Fetched {len(result)} reviews.")
    else:
        print(f"  Error: Could not find any reviews for {bank_name} ({app_id}).")
        
    return result

def main():
    all_reviews = []
    
    for bank, app_id in APPS.items():
        bank_data = scrape_bank_reviews(bank, app_id)
        all_reviews.extend(bank_data)
    
    # Save to CSV
    if all_reviews:
        df = pd.DataFrame(all_reviews)
        # Ensure directory exists
        os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
        df.to_csv(RAW_DATA_PATH, index=False)
        print(f"\nCompleted! Saved {len(df)} total reviews to {RAW_DATA_PATH}")
    else:
        print("No reviews collected.")

if __name__ == "__main__":
    main()