from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging
import transformers

# ==========================================
# 1. SERVER CONFIG & GLOBAL VARIABLES
# ==========================================
app = FastAPI(title="Safety Check API", version="1.0")

# Silence Warnings
transformers.logging.set_verbosity_error()

# Global variables to hold model in memory
model = None
tokenizer = None
device = None
MAX_LEN = 128

# Define the "Bouncer" List (Rule-Based Filter)
HARD_CUSS_WORDS = [
    
        "bsdk", "bhosdike", "bhosadike", "bhosdi", 
        "mc", "madarchod", "maderchod", 
        "bc", "bhenchod", "behenchod", "benchod",
        "chutiya", "choot", "chu", "gandu", "gaandu", 
        "lodu", "laude", "lawde", "loda",
        "randi", "chinnal", "kameena", "harami", "haramkhor",
        "tatte", "jhant", "gaand", "gand" , "lund", "chudai","chodu" ,"randikhana", "rand" , "madar"   
]

# ==========================================
# 2. STARTUP EVENT (Loads Model Once)
# ==========================================
@app.on_event("startup")
async def load_model():
    global model, tokenizer, device
    print("🚀 Server starting... Loading Model...")
    
    # 1. Detect Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Apple M2 GPU Detected (MPS)")
    else:
        device = torch.device("cpu")
        print("⚠️ Using CPU")

    # 2. Load Model & Tokenizer
    model_path = "./safety_model_v1" # Ensure this folder exists!
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, clean_up_tokenization_spaces=True)
        model.to(device)
        model.eval()
        print("✅ Model Loaded Successfully!")
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Could not load model. {e}")
        print("   Did you run 'model.py' to train it first?")

# ==========================================
# 3. REQUEST BODY FORMAT
# ==========================================
class SafetyRequest(BaseModel):
    text: str

# ==========================================
# 4. API ENDPOINT
# ==========================================
@app.post("/analyze")
async def analyze_text(request: SafetyRequest):
    """
    Input: {"text": "This place is unsafe"}
    Output: {"is_safe": false, "reason": "AI_DETECTION", "score": 0.99}
    """
    global model, tokenizer, device
    
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    text_input = request.text
    text_lower = text_input.lower()

    # --- STEP 1: THE BOUNCER (Rule Check) ---
    for word in HARD_CUSS_WORDS:
        if word in text_lower.split():
            return {
                "is_safe": False,
                "reason": "PROFANITY_FILTER",
                "flagged_word": word
            }

    # --- STEP 2: THE JUDGE (AI Check) ---
    # Tokenize
    encoding = tokenizer.encode_plus(
        text_input,
        add_special_tokens=True,
        max_length=MAX_LEN,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # Predict
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        # Get probabilities (Safe vs Unsafe)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        # Assuming Index 1 is Unsafe, Index 0 is Safe
        unsafe_score = probs[0][1].item()
        prediction_index = torch.argmax(probs, dim=1).item()

    # --- STEP 3: BOOLEAN MAPPING ---
    # Label 1 = Unsafe => is_safe = False
    # Label 0 = Safe   => is_safe = True
    
    if prediction_index == 1:
        return {
            "is_safe": False,
            "reason": "AI_CONTEXT_DETECTION",
            "confidence_score": round(unsafe_score, 4)
        }
    else:
        return {
            "is_safe": True,
            "reason": "SAFE",
            "confidence_score": round(1 - unsafe_score, 4) # Confidence in being safe
        }

# Run with: uvicorn server:app --reload