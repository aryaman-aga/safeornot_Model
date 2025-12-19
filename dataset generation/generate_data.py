import pandas as pd
import random
import os

def generate_guaranteed_hinglish():
    print("🚀 Generating Synthetic Hinglish Data...")

    # ==========================================
    # 1. VOCABULARY (The "Indian" Context)
    # ==========================================
    
    # 0 = Safe Words (Positive/Neutral)
    safe_vocab = [
        "bhai", "dost", "yaar", "sir", "mam", "uncle", "aunty", "police", 
        "security", "guards", "log", "public", "crowd", "family", "bachhe",
        "dukan", "khana", "rasta", "metro", "auto", "lighting", "mahaul"
    ]
    
    safe_adjectives = [
        "mast", "badhiya", "sahi", "safe", "acha", "helpful", "clean", 
        "shant", "crowded", "active", "supportive", "nice", "best"
    ]

    # 1 = Unsafe Words (Toxic/Threats/Slang)
    # Essential for training the model to catch harassment
    unsafe_vocab = [
        "pagal", "kutta", "kamina", "saala", "chutiya", "madarchod", 
        "bhenchod", "haramkhor", "gandu", "bsdk", "mc", "bc", "randi",
        "tharki", "creep", "chapri", "bewda", "sharaabi"
    ]
    
    unsafe_actions = [
        "marunga", "ched raha tha", "follow kar raha tha", "ghoor raha hai", 
        "comment pass kiya", "touch kiya", "pakad liya", "darata hai", 
        "loot liya", "gali di", "peeche pada hai"
    ]

    # ==========================================
    # 2. SENTENCE TEMPLATES
    # ==========================================
    
    data = []
    
    # GENERATE 1500 SAFE ROWS
    for _ in range(1500):
        # Template 1: "Bhai, [Place/Thing] [Adjective] hai"
        if random.random() < 0.5:
            text = f"{random.choice(safe_vocab)} yaha ka {random.choice(['mahaul', 'area', 'park'])} bohot {random.choice(safe_adjectives)} hai."
        # Template 2: Simple statement
        else:
            text = f"{random.choice(safe_vocab)} log bohot {random.choice(safe_adjectives)} hai yaha."
            
        data.append([text, 0]) # Label 0 = Safe

    # GENERATE 1500 UNSAFE ROWS
    for _ in range(1500):
        # Template 1: Direct Abuse "[Slang] [Action]"
        if random.random() < 0.5:
            text = f"Wo {random.choice(unsafe_vocab)} mujhe {random.choice(unsafe_actions)}."
        # Template 2: Warning "[Word] hai waha"
        else:
            text = f"Waha mat jana, bohot {random.choice(unsafe_vocab)} log hai, {random.choice(unsafe_actions)}."
            
        data.append([text, 1]) # Label 1 = Unsafe

    df_hinglish = pd.DataFrame(data, columns=['text', 'label'])
    print(f"✅ Generated {len(df_hinglish)} Hinglish rows.")

    # ==========================================
    # 3. MERGE WITH YOUR ENGLISH DATA
    # ==========================================
    try:
        # Load the file you showed me (the one with English comments)
        # Note: If your file is named differently, change this line!
        df_english = pd.read_csv("hinglish_safety_dataset.csv") 
        
        # Ensure it has the right columns
        if 'tweet' in df_english.columns:
             df_english.rename(columns={'tweet': 'text', 'class': 'label'}, inplace=True)
        
        df_english = df_english[['text', 'label']]
        
        print(f"🔄 Found {len(df_english)} existing English rows.")
        
    except Exception as e:
        print(f"⚠️ Could not load existing file: {e}")
        print("   Creating new file with ONLY Hinglish data.")
        df_english = pd.DataFrame(columns=['text', 'label'])

    # Combine
    full_df = pd.concat([df_english, df_hinglish], ignore_index=True)
    
    # SHUFFLE (Crucial so you don't just see English at the top)
    full_df = full_df.sample(frac=1).reset_index(drop=True)
    
    # Save as FINAL
    full_df.to_csv("final_safety_dataset_2.csv", index=False)
    
    print("\n" + "="*40)
    print(f"🎉 DATASET FIXED: {len(full_df)} Rows Total")
    print(f"📂 Saved to: final_safety_dataset.csv")
    print("   (Contains both Real English + Generated Hinglish)")
    print("="*40)
    print(full_df.head(10)) # Show preview to prove it worked

if __name__ == "__main__":
    generate_guaranteed_hinglish()