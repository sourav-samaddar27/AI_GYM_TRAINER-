from huggingface_hub import snapshot_download
import os

model_id = "microsoft/Phi-3-mini-4k-instruct"
save_directory = "phi3-mini-local"

# Ensure the target directory exists
os.makedirs(save_directory, exist_ok=True)

print(f"--- Starting complete download of '{model_id}' ---")
print(f"This will save ALL necessary files to '{save_directory}'.")
print("This may take a few minutes...")

try:
    # Use snapshot_download to get the entire repository
    snapshot_download(
        repo_id=model_id,
        local_dir=save_directory,
        local_dir_use_symlinks=False # Set to False to copy files directly
    )

    print("\n-------------------------------------------------")
    print("SUCCESS: Complete model repository downloaded.")
    print("You can now run sanity_check.py again.")
    print("-------------------------------------------------")

except Exception as e:
    print(f"\nERROR: An error occurred during the complete download: {e}")