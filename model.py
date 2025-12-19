import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm 
import os
import logging
import transformers

# 1. DISABLE TOKENIZER PARALLELISM (Fixes the warning and deadlock risk)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Silence HuggingFace warnings
transformers.logging.set_verbosity_error()

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_NAME = "google/muril-base-cased"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 1e-5

def train_model():
    # 2. SETUP DEVICE
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"✅ Apple M2 GPU Detected: Using {device}")
    else:
        device = torch.device("cpu")
        print("⚠️ MPS not found. Using CPU")

    # 3. LOAD DATA
    filename = "final_safety_dataset.csv" # Ensure this matches your file
    if not os.path.exists(filename):
        print(f"❌ Error: {filename} not found.")
        return

    print("⏳ Loading CSV...")
    df = pd.read_csv(filename) 
    
    # === SPEED FIX: Take a random 20k sample ===
    # This keeps the variety but cuts training time from 1 hour to 15 mins
    if len(df) > 20000:
        print(f"✂️  Dataset too big for laptop ({len(df)} rows). Sampling 20,000 rows...")
        df = df.sample(n=20000, random_state=42).reset_index(drop=True)
    # ===========================================

    print(f"   Training on {len(df)} rows.")
    
    # 4. PRE-TOKENIZATION (The Speed Boost)
    print("⚡️ Pre-tokenizing all data (This takes 30-60 seconds)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, clean_up_tokenization_spaces=True)
    
    # Split text and labels
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].astype(str).tolist(), 
        df['label'].tolist(), 
        test_size=0.1, 
        random_state=42
    )

    # Tokenize in BULK (Much faster than doing it one-by-one in loop)
    tokens_train = tokenizer.batch_encode_plus(
        train_texts,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_token_type_ids=False,
        return_tensors='pt' # Return PyTorch tensors directly
    )
    
    tokens_val = tokenizer.batch_encode_plus(
        val_texts,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_token_type_ids=False,
        return_tensors='pt'
    )

    # 5. CREATE TENSOR DATASETS (Very fast to load)
    train_seq = tokens_train['input_ids']
    train_mask = tokens_train['attention_mask']
    train_y = torch.tensor(train_labels)

    val_seq = tokens_val['input_ids']
    val_mask = tokens_val['attention_mask']
    val_y = torch.tensor(val_labels)

    train_data = TensorDataset(train_seq, train_mask, train_y)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

    val_data = TensorDataset(val_seq, val_mask, val_y)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=BATCH_SIZE)

    # 6. MODEL SETUP
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    print(f"🚀 Starting Training...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        # Progress bar
        loop = tqdm(train_dataloader, leave=True)
        
        for batch in loop:
            # Push batch to GPU
            batch = [r.to(device) for r in batch]
            sent_id, mask, labels = batch

            model.zero_grad()
            
            outputs = model(sent_id, attention_mask=mask, labels=labels)
            
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            
            loop.set_description(f"Epoch {epoch+1}")
            loop.set_postfix(loss=loss.item())
        
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} Completed | Avg Loss: {avg_loss:.4f}")

    # Save
    print("💾 Saving model...")
    model.save_pretrained("./safety_model_v1")
    tokenizer.save_pretrained("./safety_model_v1")
    print("🎉 Model Saved Successfully!")

def check_safety(text_input):
    # 1. THE "BOUNCER" (Strict Rule-Based Check)
    # If any of these words appear, mark Unsafe immediately.
    # Note: Do NOT put dual-meaning words like 'kutta' (dog/insult) or 'saala' (brother-in-law/insult) here.
    # Let the AI handle those. Put only PURE cuss words here.
    hard_cuss_words = [
        "bsdk", "bhosdike", "bhosadike", "bhosdi", 
        "mc", "madarchod", "maderchod", 
        "bc", "bhenchod", "behenchod", "benchod",
        "chutiya", "choot", "chu", "gandu", "gaandu", 
        "lodu", "laude", "lawde", "loda",
        "randi", "chinnal", "kameena", "harami", "haramkhor",
        "tatte", "jhant", "gaand", "gand" , "lund", "chudai","chodu" ,"randikhana"
    ]
    
    # Normalize text (lowercase) for matching
    text_lower = text_input.lower()
    
    # Check if any cuss word is in the text
    for word in hard_cuss_words:
        # We check with spaces to avoid matching substrings incorrectly 
        # (e.g., don't flag "class" because it has "ass")
        # But for Hinglish 'bsdk', direct containment is usually fine.
        if word in text_lower.split(): 
            return "No (Unsafe) [Detected: Rule-Based Cuss Word]"

    # ---------------------------------------------------------
    # 2. THE "JUDGE" (Deep Learning Context Check)
    # If no cuss words found, ask the model.
    # ---------------------------------------------------------
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    path = "./safety_model_v1"
    try:
        model = AutoModelForSequenceClassification.from_pretrained(path)
        tokenizer = AutoTokenizer.from_pretrained(path, clean_up_tokenization_spaces=True)
    except:
        return "Model not found. Train first."
    
    model.to(device)
    model.eval()

    encoding = tokenizer.encode_plus(
        text_input, add_special_tokens=True, max_length=MAX_LEN,
        return_token_type_ids=False, padding='max_length', truncation=True,
        return_attention_mask=True, return_tensors='pt',
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        prediction = torch.argmax(outputs.logits, dim=1).item()

    return "No (Unsafe) [Detected: AI Context]" if prediction == 1 else "Yes (Safe)"

if __name__ == "__main__":
    # train_model() 
    
    print("\n--- Model Ready ---")
    print("Test 1: 'she is fighting for her rights bsdk' ->", check_safety("she is fighting for her rights bsdk"))
    print("Test 2: 'Hey everyone had a pretty bad bad experience at Delhi, someone harassed me' ->", check_safety(" Hey everyone had a pretty bad bad experience at Delhi, someone harassed me"))
    print("Test 3: 'ladhkiyo ko harass karna chaiye' ->", check_safety(" ladhkiyo ko harass karna chaiye"))