import os
import pandas as pd
import kagglehub

def stage_dataset():
    print("🚀 Starting Step 4: Data Fetch & Staging Pipeline...")
    
    # 1. Download the latest version from Kaggle programmatically
    download_path = kagglehub.dataset_download("clmentbisaillon/fake-and-real-news-dataset")
    print(f"📥 Dataset downloaded to local cache at: {download_path}")

    # 2. Point to the individual CSV files inside the cache
    true_file = os.path.join(download_path, "True.csv")
    fake_file = os.path.join(download_path, "Fake.csv")

    # 3. Read the data streams using pandas
    print("📖 Reading raw data streams...")
    true_df = pd.read_csv(true_file)
    fake_df = pd.read_csv(fake_file)

    # 4. Inject target class labels (1 for Real news, 0 for Fake news)
    true_df['label'] = 1
    fake_df['label'] = 0

    # 5. Create local directories structurally if they don't exist yet
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    # 6. Save untouched copies into your local raw directory
    print("📂 Archiving individual sets to 'data/raw/'...")
    true_df.to_csv("data/raw/True.csv", index=False)
    fake_df.to_csv("data/raw/Fake.csv", index=False)

    # 7. Concatenate and uniformly shuffle the entries
    print("🔄 Combining and random-shuffling datasets...")
    combined_df = pd.concat([true_df, fake_df], ignore_index=True)
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 8. Save the unified dataset to the processed folder
    processed_output_path = "data/processed/combined_news.csv"
    combined_df.to_csv(processed_output_path, index=False)
    print(f"✅ Unified data stream compiled successfully at: {processed_output_path}")

    # 9. Pipeline validation report
    print("\n📊 --- Staging Verification Metrics ---")
    print(f"Total Rows Compiled: {combined_df.shape[0]}")
    print(f"Features Detected: {combined_df.shape[1]} -> {combined_df.columns.tolist()}")
    print(f"Real News Articles (Label 1): {combined_df[combined_df['label'] == 1].shape[0]}")
    print(f"Fake News Articles (Label 0): {combined_df[combined_df['label'] == 0].shape[0]}")

if __name__ == "__main__":
    stage_dataset()