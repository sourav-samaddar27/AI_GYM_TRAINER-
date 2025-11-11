# fine_tune_gemma.py (Final version with a 2-step loading process to bypass the library bug)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import load_dataset
import os

# --- 1. Configuration ---
BASE_MODEL_ID = "gemma-2b-it-local"
DATA_FOLDER = "cleaned data"
NEW_ADAPTER_NAME = "gemma-gym-coach-v1"

# --- Gemma Chat Template Formatting Function ---
def format_gemma_chat_template(example):
    return {
        "text": f"<start_of_turn>user\n{example['instruction']}<end_of_turn>\n<start_of_turn>model\n{example['response']}<end_of_turn>"
    }

def main():
    print(f"--- Starting QLoRA Fine-Tuning for '{BASE_MODEL_ID}' ---")
    print("--- Using a 2-step loading process to bypass the adapter check bug ---")

    # --- 2. Load and Format the Dataset ---
    print(f"Loading all CSV files from the '{DATA_FOLDER}' directory...")
    all_csv_files = os.path.join(DATA_FOLDER, "*.csv")
    data = load_dataset("csv", data_files=all_csv_files)
    print("Dataset loaded successfully. Now formatting for Gemma chat template...")
    formatted_data = data.map(format_gemma_chat_template)
    print("Dataset formatted successfully.")

    # --- 3. Load the Base Model & Tokenizer (The Definitive Fix) ---

    # Step 3.1: Load the model from local files WITHOUT quantization first.
    # This avoids the buggy preliminary adapter check.
    print(f"Loading base model '{BASE_MODEL_ID}' from local files...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        local_files_only=True,
        torch_dtype=torch.bfloat16 # Load in a memory-efficient format
    )
    print("Base model loaded successfully.")
    
    # Step 3.2: Now that the model is loaded, we apply quantization.
    # We must re-initialize the model with the quantization config. This seems redundant
    # but it's the correct way to apply quantization to an already-loaded model.
    print("Applying 4-bit quantization to the loaded model...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=quantization_config,
        local_files_only=True
    )
    print("Quantization applied successfully.")
    
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID,
        local_files_only=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded successfully.")

    # --- 4. Configure PEFT (LoRA) ---
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    print("PEFT (LoRA) configured successfully.")
    model.print_trainable_parameters()

    # --- 5. Set Up the Trainer ---
    # This section remains the same
    def collate_fn(batch):
        texts = [item['text'] for item in batch]
        tokens = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
        tokens['labels'] = tokens['input_ids'].clone()
        return tokens

    training_args = TrainingArguments(
        output_dir="./training_.output",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        max_steps=300,
        logging_steps=10,
        save_steps=50,
        fp16=False,
        bf16=False,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=formatted_data['train'],
        data_collator=collate_fn,
    )

    print("\n--- Starting Training ---")
    trainer.train()
    print("--- Training Finished ---")

    # --- 6. Save the Final Adapter ---
    print(f"Saving the fine-tuned adapter to '{NEW_ADAPTER_NAME}'...")
    model.save_pretrained(NEW_ADAPTER_NAME)
    print("Adapter saved successfully.")


if __name__ == "__main__":
    main()