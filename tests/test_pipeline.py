import unittest
import pandas as pd
from textblob import TextBlob

# logic from your analysis script
def get_sentiment(text):
    return TextBlob(str(text)).sentiment.polarity

# logic from your cleaner script
def clean_date(date_str):
    return pd.to_datetime(date_str).strftime('%Y-%m-%d')

class TestDataPipeline(unittest.TestCase):

    def test_sentiment_analysis(self):
        """Test if sentiment analysis correctly identifies positive/negative text"""
        positive_text = "I love this app, it is amazing!"
        negative_text = "This app crashes every time, terrible."
        
        pos_score = get_sentiment(positive_text)
        neg_score = get_sentiment(negative_text)
        
        self.assertGreater(pos_score, 0, "Positive text should have score > 0")
        self.assertLess(neg_score, 0, "Negative text should have score < 0")

    def test_date_normalization(self):
        """Test if dates are correctly normalized to YYYY-MM-DD"""
        input_date = "2025-11-30 14:30:00"
        expected_date = "2025-11-30"
        
        cleaned = clean_date(input_date)
        self.assertEqual(cleaned, expected_date)

if __name__ == '__main__':
    unittest.main()