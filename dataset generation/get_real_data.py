import pandas as pd
from datasets import load_dataset
import numpy as np

def create_massive_dataset():
    print("🚀 Starting Massive Data Download... (This will take 2-3 minutes)")
    
    all_data = []

    # ==========================================
    # SOURCE 1: L3Cube Hinglish (Indian Context)
    # ==========================================
    # 0 = Non-Hate, 1 = Hate
    try:
        print("⬇️  Downloading L3Cube Hinglish...")
        ds = load_dataset("l3cube-pune/hinglish-code-mixed-offensive-custom", split="train")
        df = pd.DataFrame(ds)
        df = df[['text', 'label']]
        # L3Cube is already: 0=Safe, 1=Unsafe. No change needed.
        all_data.append(df)
        print(f"   ✅ Added {len(df)} rows from L3Cube.")
    except Exception as e:
        print(f"   ⚠️ Failed L3Cube: {e}")

    # ==========================================
    # SOURCE 2: TweetEval (Social Media Slang)
    # ==========================================
    # 0 = Non-offensive, 1 = Offensive
    try:
        print("⬇️  Downloading TweetEval (Offensive)...")
        ds = load_dataset("tweet_eval", "offensive", split="train")
        df = pd.DataFrame(ds)
        df = df[['text', 'label']]
        # TweetEval is already: 0=Safe, 1=Unsafe. No change needed.
        all_data.append(df)
        print(f"   ✅ Added {len(df)} rows from TweetEval.")
    except Exception as e:
        print(f"   ⚠️ Failed TweetEval: {e}")

    # ==========================================
    # SOURCE 3: Davidson Hate Speech (Complex English)
    # ==========================================
    # Original Labels: 0=Hate, 1=Offensive, 2=Neither (Safe)
    # WE NEED TO REMAP: 0&1 -> 1 (Unsafe), 2 -> 0 (Safe)
    try:
        print("⬇️  Downloading Davidson Hate Speech...")
        ds = load_dataset("hate_speech_offensive", split="train")
        df = pd.DataFrame(ds)
        df = df[['tweet', 'class']] # Columns are named differently here
        df.columns = ['text', 'label'] # Rename to match others
        
        # Remap Logic: 
        # If class is 2 (Neither), label = 0 (Safe)
        # If class is 0 or 1 (Hate/Offensive), label = 1 (Unsafe)
        df['label'] = df['label'].apply(lambda x: 0 if x == 2 else 1)
        
        all_data.append(df)
        print(f"   ✅ Added {len(df)} rows from Davidson.")
    except Exception as e:
        print(f"   ⚠️ Failed Davidson: {e}")

    # ==========================================
    # MERGE & EXPORT
    # ==========================================
    print("🔄 Merging all datasets...")
    if not all_data:
        print("❌ Error: No data downloaded.")
        return

    full_df = pd.concat(all_data, ignore_index=True)
    
    # Shuffle completely
    full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Clean text (remove newlines, keep it simple)
    full_df['text'] = full_df['text'].astype(str).str.replace('\n', ' ')

    # Save
    filename = "hinglish_safety_dataset.csv"
    full_df.to_csv(filename, index=False)
    
    print("\n" + "="*40)
    print(f"🎉 FINAL DATASET READY: {len(full_df)} Rows")
    print(f"📂 Saved to: {filename}")
    print("="*40)
    
    # Show distribution
    print("Label Distribution (0=Safe, 1=Unsafe):")
    print(full_df['label'].value_counts())

if __name__ == "__main__":
    create_massive_dataset()