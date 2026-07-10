import pandas as pd
from src.preprocessing import full_preprocess_pipeline

print("Testing custom text-cleaning pipeline...")

# Read just the first 2 articles from your combined dataset to test
df = pd.read_csv("data/processed/combined_news.csv", nrows=2)

for index, row in df.iterrows():
    print(f"\n--- Original Text (Article #{index + 1}) ---")
    print(row['text'][:150] + "...") # Show first 150 characters
    
    # Run it through our new engine
    cleaned = full_preprocess_pipeline(row['text'])
    
    print(f"\n--- Cleaned Text (Article #{index + 1}) ---")
    print(cleaned[:150] + "...")
    print("="*50)