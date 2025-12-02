import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

CSV_PATH = os.path.join("data", "sentiment_analysis.csv")

def create_plots():
    df = pd.read_csv(CSV_PATH)
    sns.set(style="whitegrid")

    # Plot 1: Sentiment by Bank
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='bank_name', hue='sentiment_label', palette='viridis')
    plt.title('Sentiment Distribution by Bank')
    plt.savefig('sentiment_by_bank.png')
    print("Saved sentiment_by_bank.png")

    # Plot 2: Rating Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='rating', bins=5, kde=True, hue='bank_name', element="step")
    plt.title('Rating Distribution Comparison')
    plt.savefig('rating_dist.png')
    print("Saved rating_dist.png")
    
    # Plot 3: Top Themes (Pain Points)
    # Filter for negative reviews only
    neg_df = df[df['sentiment_label'] == 'Negative']
    # Explode the themes (since they are comma separated)
    theme_series = neg_df['theme'].str.split(', ').explode()
    
    plt.figure(figsize=(12, 6))
    sns.countplot(y=theme_series, order=theme_series.value_counts().index, palette='rocket')
    plt.title('Top Pain Point Themes (Negative Reviews)')
    plt.savefig('pain_points.png')
    print("Saved pain_points.png")

if __name__ == "__main__":
    create_plots()