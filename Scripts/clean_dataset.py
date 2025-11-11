# clean_dataset_v2.py
import pandas as pd
import os
import glob
import nltk
import re
from nltk.corpus import words

# --- One-time setup for NLTK: Download the English dictionary
try:
    # Check if the word list is already downloaded
    english_words = set(words.words())
except LookupError:
    print("NLTK 'words' corpus not found. Downloading now...")
    nltk.download('words')
    english_words = set(words.words())
print("English dictionary is ready.")


# --- Configuration ---
SOURCE_DATA_FOLDER = "cleaned data"
DESTINATION_DATA_FOLDER = "truly_cleaned_data_v2" # A new folder for the ultra-clean files


def is_high_quality(text, min_words=4, vocab_threshold=0.7):
    """
    This is the smart filter. It checks two things:
    1. Is the text long enough to be a real sentence?
    2. Are most of the words in the text real English words?
    """
    if not isinstance(text, str):
        return False

    # Clean the text: lowercase, remove punctuation, keep only letters and spaces
    cleaned_text = re.sub(r'[^a-z\s]', '', text.lower())
    
    # Split into words
    text_words = cleaned_text.split()

    # --- Filter 1: Minimum Length ---
    if len(text_words) < min_words:
        return False

    # --- Filter 2: Vocabulary Check ---
    # Count how many of the words are in the standard English dictionary
    english_word_count = sum(1 for word in text_words if word in english_words)
    
    # Calculate the ratio of real English words
    vocab_ratio = english_word_count / len(text_words)
    
    # Only keep the text if the ratio is above our threshold (e.g., 70%)
    return vocab_ratio >= vocab_threshold


def main():
    print("--- Starting High-Quality Dataset Cleaning Process ---")
    os.makedirs(DESTINATION_DATA_FOLDER, exist_ok=True)
    print(f"Ultra-clean files will be saved in: '{DESTINATION_DATA_FOLDER}'")

    csv_files = glob.glob(os.path.join(SOURCE_DATA_FOLDER, "*.csv"))
    if not csv_files:
        print(f"FATAL ERROR: No CSV files found in '{SOURCE_DATA_FOLDER}'.")
        return

    total_rows_processed = 0
    total_rows_kept = 0

    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        print(f"\nProcessing file: {file_name}...")

        try:
            df = pd.read_csv(file_path)
            original_row_count = len(df)
            total_rows_processed += original_row_count

            # Drop rows with any missing data first
            df.dropna(subset=['instruction', 'response'], inplace=True)
            
            # --- Apply the new, smart quality filter ---
            clean_df = df[df.apply(lambda row: is_high_quality(row['instruction']) and is_high_quality(row['response']), axis=1)]

            final_row_count = len(clean_df)
            total_rows_kept += final_row_count
            
            print(f"  > Original rows: {original_row_count}")
            print(f"  > Rows after high-quality filtering: {final_row_count} ({((original_row_count - final_row_count) / original_row_count):.1%} removed)")

            if not clean_df.empty:
                destination_path = os.path.join(DESTINATION_DATA_FOLDER, file_name)
                clean_df.to_csv(destination_path, index=False)
                print(f"  > Saved ultra-clean file to: {destination_path}")

        except Exception as e:
            print(f"  > ERROR processing {file_name}: {e}")

    print("\n--- Cleaning Process Finished ---")
    print(f"Total rows processed: {total_rows_processed}")
    print(f"Total high-quality rows kept: {total_rows_kept}")
    if total_rows_processed > 0:
        print(f"Overall reduction: {((total_rows_processed - total_rows_kept) / total_rows_processed):.1%}")

if __name__ == "__main__":
    main()