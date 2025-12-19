# SafeOrNot – Context-Aware Safety Classification API

SafeOrNot is a hybrid rule-based and machine learning safety detection system designed to classify text as Safe or Unsafe, with a strong focus on Indian and Hinglish language contexts.  
The project exposes a FastAPI-based inference service suitable for moderation, safety scoring, and abuse detection use cases.

---

## Features

- Context-aware text safety classification using a Transformer-based model
- Rule-based profanity filtering for immediate hard violations
- Hinglish and Indian slang handling
- FastAPI backend for real-time inference
- Hugging Face–hosted model weights (no large files in GitHub)
- CPU and Apple Silicon (MPS) support

---

## Project Architecture


safeornot_Model/
├── server.py # FastAPI inference server
├── model.py # Training / experimentation logic
├── dataset_generation/ # Dataset preparation scripts
├── *.csv # Training and evaluation datasets
├── requirements.txt
├── README.md


Model weights are intentionally excluded from this repository.

---

## Model Weights

The trained model is hosted on Hugging Face.

Model URL:  
https://huggingface.co/aryaman1222/safeornot-safety-model

The API automatically downloads the model at runtime using Hugging Face’s caching mechanism.  
No manual placement of model files is required.

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/aryaman-aga/safeornot_Model.git
cd safeornot_Model


2. Create and Activate Virtual Environment
python3 -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate    # Windows

3. Install Dependencies
pip install -r requirements.txt

4. Run the API Server
uvicorn server:app --reload


On first run, the model will be downloaded from Hugging Face and cached locally.

API Usage
Interactive API Documentation

Open in a browser:

http://127.0.0.1:8000/docs

Endpoint: /analyze

Request body:

{
  "text": "Yeh jagah bilkul unsafe lag rahi hai"
}


Response (Unsafe):

{
  "is_safe": false,
  "reason": "AI_CONTEXT_DETECTION",
  "confidence_score": 0.94
}


Response (Safe):

{
  "is_safe": true,
  "reason": "SAFE",
  "confidence_score": 0.91
}

Detection Logic

Rule-Based Filter
Immediate rejection when hard profanity is detected.
This prevents obvious unsafe content from reaching the ML model.

Machine Learning Context Detection
Transformer-based sequence classification handles contextual abuse, implicit harassment, and Hinglish phrasing.

Hardware Support

CPU (default)

Apple Silicon GPU via PyTorch MPS

Automatic device detection at startup

Intended Use Cases

Community moderation systems

Safety scoring for review or location-based platforms

Abuse and harassment detection pipelines

Educational and research projects in NLP safety

Disclaimer

This project is intended for research and educational purposes.
Predictions should be validated before use in high-stakes or production-critical systems.

Author

Aryaman Agarwal
B.Tech Computer Science
