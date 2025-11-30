# Customer Experience Analytics for Fintech Apps

## Project Overview
This project scrapes and analyzes Google Play Store reviews for three Ethiopian banks (CBE, BOA, Dashen) to identify customer sentiment and key themes.

## Folder Structure
- `src/`: Python scripts for scraping, cleaning, and analysis.
- `data/`: CSV files (Raw and Processed).
- `requirements.txt`: Project dependencies.

## Setup Instructions
1. Create a virtual environment: `python -m venv venv`
2. Activate it and install dependencies: `pip install -r requirements.txt`
3. Run the pipeline:
   ```bash
   python src/scraper.py
   python src/cleaner.py
   python src/analysis.py
   ```
Methodology
- Scraping: Uses google-play-scraper to fetch 450+ reviews per bank.
- Cleaning: Removes duplicates, handles nulls, and normalizes dates.
- Analysis: Uses TextBlob for sentiment scoring and Scikit-Learn (TF-IDF) for keyword extraction.