import pandas as pd
import time
from src.preprocessing import full_preprocess_pipeline

def run_mass_preprocessing():
    print("[Preprocessing] Loading combined dataset...")
    df = pd.read_csv("data/processed/combined_news.csv")
    
    # Drop any rows where the text column is completely missing/empty
    df = df.dropna(subset=['text']).reset_index(drop=True)
    
    print(f"[Preprocessing] Cleaning and tokenizing {len(df)} articles. This will take around 30-45 seconds...")
    start_time = time.time()
    
    # Apply your custom cleaning pipeline to the text column
    df['clean_text'] = df['text'].apply(full_preprocess_pipeline)
    
    # Drop any rows that became completely empty strings after stripping stopwords
    df = df[df['clean_text'].str.strip() != ""].reset_index(drop=True)
    
    end_time = time.time()
    print(f"[Preprocessing] Preprocessing complete! Time taken: {end_time - start_time:.2f} seconds.")
    
    # Save the polished output to your processed data directory
    output_path = "data/processed/cleaned_news.csv"
    combined_path = "data/processed/combined_news.csv"
    df[['clean_text', 'label']].to_csv(output_path, index=False)
    df.to_csv(combined_path, index=False)
    print(f"[Preprocessing] Cleaned dataset successfully saved to: {output_path} and {combined_path}")

if __name__ == "__main__":
    run_mass_preprocessing()
