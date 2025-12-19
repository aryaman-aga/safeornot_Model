from huggingface_hub import HfApi

api = HfApi()

api.upload_folder(
    folder_path="safety_model_v1",
    repo_id="aryaman1222/safeornot-safety-model",
    repo_type="model",
)

print("✅ Model uploaded successfully")
