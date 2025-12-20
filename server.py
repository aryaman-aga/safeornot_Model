from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import re
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
import os

# ==========================================
# CONFIG
# ==========================================
app = FastAPI(title="Safety Check API", version="1.0")

MODEL_REPO = "aryaman1222/safeornot-safety-model-onnx"
MAX_LEN = 128

tokenizer = None
ort_session = None

# Profanity list
HARD_CUSS_WORDS = [
    "bsdk", "bhosdike", "bhosadike", "bhosdi",
    "mc", "madarchod", "maderchod",
    "bc", "bhenchod", "behenchod", "benchod",
    "chutiya", "choot", "chu", "gandu", "gaandu",
    "lodu", "laude", "lawde", "loda",
    "randi", "chinnal", "kameena", "harami", "haramkhor",
    "tatte", "jhant", "gaand", "gand", "lund",
    "chudai", "chodu", "randikhana", "rand", "madar"
]

# ==========================================
# STARTUP
# ==========================================
@app.on_event("startup")
async def load_model():
    global tokenizer, ort_session

    print("🚀 Loading ONNX model...")

    try:
        # Check if we have a local model path (from Docker)
        local_model_path = os.getenv("MODEL_PATH")
        
        if local_model_path and os.path.exists(local_model_path):
            print(f"📂 Loading from local path: {local_model_path}")
            tokenizer = AutoTokenizer.from_pretrained(local_model_path, use_fast=True)
            onnx_path = os.path.join(local_model_path, "model.onnx")
        else:
            print(f"☁️  Downloading from Hugging Face: {MODEL_REPO}")
            tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO, use_fast=True)
            onnx_path = hf_hub_download(
                repo_id=MODEL_REPO,
                filename="model.onnx",
                token=os.getenv("HF_TOKEN")
            )

        # Create ONNX Runtime session
        ort_session = ort.InferenceSession(
            onnx_path,
            providers=["CPUExecutionProvider"]
        )

        # Warm-up (important)
        dummy = tokenizer(
            "warmup",
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="np"
        )

        # Fix: Include token_type_ids if the tokenizer produces them
        inputs = {
            "input_ids": dummy["input_ids"],
            "attention_mask": dummy["attention_mask"]
        }
        if "token_type_ids" in dummy:
            inputs["token_type_ids"] = dummy["token_type_ids"]

        ort_session.run(None, inputs)

        print("✅ ONNX Runtime ready")

    except Exception as e:
        raise RuntimeError(f"Failed to load ONNX model: {e}")

# ==========================================
# REQUEST SCHEMA
# ==========================================
class SafetyRequest(BaseModel):
    text: str

# ==========================================
# ENDPOINT
# ==========================================
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": ort_session is not None}

@app.post("/analyze")
async def analyze_text(request: SafetyRequest):

    if ort_session is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty text")

    text_lower = text.lower()

    # Rule-based filter
    tokens = re.findall(r"\b\w+\b", text_lower)
    for word in HARD_CUSS_WORDS:
        if any(word in token for token in tokens):
            return {
                "is_safe": False,
                "reason": "PROFANITY_FILTER",
                "flagged_word": word
            }

    # Tokenize
    encoding = tokenizer(
        text,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="np"
    )

    inputs = {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"]
    }
    
    # Fix: Include token_type_ids for inference as well
    if "token_type_ids" in encoding:
        inputs["token_type_ids"] = encoding["token_type_ids"]

    logits = ort_session.run(None, inputs)[0]

    # Softmax
    exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp / np.sum(exp, axis=1, keepdims=True)

    unsafe_score = float(probs[0][1])
    pred = int(np.argmax(probs, axis=1)[0])

    if pred == 1:
        return {
            "is_safe": False,
            "reason": "AI_CONTEXT_DETECTION",
            "confidence_score": round(unsafe_score, 4)
        }
    else:
        return {
            "is_safe": True,
            "reason": "SAFE",
            "confidence_score": round(1 - unsafe_score, 4)
        }
