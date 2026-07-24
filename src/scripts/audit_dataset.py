import pandas as pd
import re
from collections import Counter

def audit():
    print("--- Dataset Source Audit ---")
    combined_path = "data/processed/combined_news.csv"
    if not pd.io.common.file_exists(combined_path):
        print(f"Error: {combined_path} not found.")
        return
        
    df = pd.read_csv(combined_path)
    print(f"Total articles: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check subjects
    if 'subject' in df.columns:
        print("\nSubject Distribution by Label (1=Real, 0=Fake):")
        print(df.groupby(['label', 'subject']).size())
        
    # Analyze text for publisher mentions
    # e.g., Reuters, Associated Press, AP, Infowars, New York Times, etc.
    reuters_pattern = re.compile(r'\breuters\b', re.IGNORECASE)
    ap_pattern = re.compile(r'\bassociated press\b|\b\(ap\)\b', re.IGNORECASE)
    
    reuters_real = df[df['label'] == 1]['text'].apply(lambda x: bool(reuters_pattern.search(str(x)))).sum()
    reuters_fake = df[df['label'] == 0]['text'].apply(lambda x: bool(reuters_pattern.search(str(x)))).sum()
    
    ap_real = df[df['label'] == 1]['text'].apply(lambda x: bool(ap_pattern.search(str(x)))).sum()
    ap_fake = df[df['label'] == 0]['text'].apply(lambda x: bool(ap_pattern.search(str(x)))).sum()
    
    print(f"\nReuters mentions in Real: {reuters_real} / {len(df[df['label']==1])}")
    print(f"Reuters mentions in Fake: {reuters_fake} / {len(df[df['label']==0])}")
    print(f"AP/Associated Press mentions in Real: {ap_real} / {len(df[df['label']==1])}")
    print(f"AP/Associated Press mentions in Fake: {ap_fake} / {len(df[df['label']==0])}")
    
    # Let's check the start of the text for "(Reuters)" or similar datelines
    dateline_reuters = df[df['label'] == 1]['text'].apply(lambda x: str(x).strip().startswith("WASHINGTON (Reuters)") or "Reuters" in str(x)[:50]).sum()
    print(f"Reuters signature in first 50 chars of Real: {dateline_reuters} / {len(df[df['label']==1])}")

if __name__ == "__main__":
    audit()
