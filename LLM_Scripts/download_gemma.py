# download_gemma.py
from huggingface_hub import snapshot_download
import os

model_id = "google/gemma-2b-it"
save_directory = "google/gemma-2b-it"
os.makedirs(save_directory, exist_ok=True)

print(f"--- Starting smart download of '{model_id}' ---")
snapshot_download(
    repo_id=model_id,
    local_dir=save_directory,
    local_dir_use_symlinks=False,
    ignore_patterns=["*.gguf*"]
)
print("\n--- SUCCESS: Base model downloaded. ---")