import pandas as pd
import random

def generate_indian_context():
    print("🚀 Generating Pure Indian Safety Dataset...")

    # ==========================================
    # 1. TOXIC INDIAN VOCABULARY (The stuff it missed)
    # ==========================================
    # We include variations of spellings common in India
    toxic_words = [
        "bsdk", "bhosdike", "mc", "bc", "bhenchod", "madarchod", 
        "chutiya", "gandu", "kamina", "kutta", "saala", "harami",
        "randi", "chinnal", "item", "maal", "pataka", "tota",
        "tharki", "creep", "rapist", "molester"
    ]
    
    toxic_intents = [
        "kya kar raha hai", "bahar mil tu", "number de apna", 
        "rate kya hai", "aaja gaadi mein baith", "uthwa lunga",
        "chehra kharab kar dunga", "follow kar raha hai",
        "gandi nazron se dekh raha tha", "zabardasti kar raha tha",
        "touch karne ki koshish ki", "harassment kar raha hai",
        "ganda comment pass kiya"
    ]

    # ==========================================
    # 2. SAFE VOCABULARY
    # ==========================================
    safe_words = [
        "bhai", "bhaiya", "uncle", "aunty", "dost", "sir", "madam",
        "police", "guard", "security", "driver", "shopkeeper"
    ]
    
    safe_contexts = [
        "ne help ki", "bohot ache hain", "safe hai ye jagah",
        "time par drop kar diya", "thank you bola", "guide kiya rasta",
        "crowded area hai", "cctv laga hai", "lighting achi hai",
        "family ke saath safe hai", "koi darne ki baat nahi",
        "police patrolling hoti hai", "harassment ke khilaf laws hain"
    ]

    # ==========================================
    # 3. GENERATION LOGIC (10,000 Rows)
    # ==========================================
    data = []
    
    for _ in range(10000):
        # --- GENERATE UNSAFE (Label 1) ---
        # Structure A: Slang + Intent ("Bsdk kya kar raha hai")
        if random.random() < 0.5:
            text = f"{random.choice(toxic_words)} {random.choice(toxic_intents)}"
        # Structure B: Just Intent ("Harassment kar raha tha")
        else:
            text = f"Waha mat jana, ladke {random.choice(toxic_intents)}"
        
        data.append([text, 1])

    for _ in range(10000):
        # --- GENERATE SAFE (Label 0) ---
        # Structure A: Safe Person + Action ("Guard ne help ki")
        if random.random() < 0.5:
            text = f"{random.choice(safe_words)} {random.choice(safe_contexts)}"
        # Structure B: General Safety ("Safe hai ye jagah")
        else:
            text = f"Ye area bilkul {random.choice(safe_contexts)}"
            
        data.append([text, 0])

    # ==========================================
    # 4. ADD SPECIFIC "TRAP" CASES (Your failed examples)
    # ==========================================
    # We force these specific phrases to be in the training data
    hardcoded_unsafe = [
        "bsdk kya kar raha hai tu",
        "harassment hona chaiye",
        "ladki chedna maza hai",
        "tu bahar mil batata hu",
        "item mast hai",
        "maal kaisa hai",
        "chutiya hai kya",
        "mc bc kar raha tha",
        "harassing is fun", # English Trap
        "rape threats de raha tha"
    ]
    
    for text in hardcoded_unsafe:
        # Add them 10 times each so the model REALLY learns them
        for _ in range(10):
            data.append([text, 1])

    # ==========================================
    # 5. SAVE
    # ==========================================
    df = pd.DataFrame(data, columns=['text', 'label'])
    df = df.sample(frac=1).reset_index(drop=True) # Shuffle
    
    filename = "indian_safety_final.csv"
    df.to_csv(filename, index=False)
    
    print(f"🎉 Generated {len(df)} rows of PURE INDIAN context.")
    print(f"📂 Saved to: {filename}")

if __name__ == "__main__":
    generate_indian_context()