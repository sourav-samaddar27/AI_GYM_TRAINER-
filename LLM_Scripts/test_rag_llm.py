# ==============================================================================
# FINAL test_rag_llm.py (Definitive, Correct Response Parsing)
# ==============================================================================
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, LoraConfig
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os, warnings, json
import numpy as np

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- 1. CONFIGURATION ---
BASE_MODEL_ID = "google/gemma-2b-it"
ADAPTER_PATH = "gemma_gym_coach_v2"
PORTABLE_DB_PATH = "portable_db"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

def main():
    print("--- AI Gym Coach (Final Mac Version) ---")
    
    # --- 2. LOAD AND REBUILD RAG DATABASE LOCALLY ---
    print("\n--- Loading portable RAG components and building FAISS index in memory... ---")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={'device': 'cpu'})
    
    text_chunks_path = os.path.join(PORTABLE_DB_PATH, "texts.json")
    embeddings_path = os.path.join(PORTABLE_DB_PATH, "embeddings.npy")
    with open(text_chunks_path, 'r') as f:
        text_chunks = json.load(f)
    numpy_embeddings = np.load(embeddings_path)
    
    text_embedding_pairs = list(zip(text_chunks, numpy_embeddings.tolist()))
    
    db = FAISS.from_embeddings(text_embedding_pairs, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 5})
    print("--- RAG components loaded and index built successfully. ---")

    # --- 3. LOAD MODEL & ADAPTER ---
    device = torch.device("cpu")
    print(f"\n--- Loading base model and adapter... ---")
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    
    adapter_config_path = os.path.join(ADAPTER_PATH, "adapter_config.json")
    with open(adapter_config_path, 'r') as f:
        config_dict = json.load(f)
    config = LoraConfig(
        r=config_dict.get("r"), lora_alpha=config_dict.get("lora_alpha"),
        target_modules=config_dict.get("target_modules"), lora_dropout=config_dict.get("lora_dropout"),
        bias=config_dict.get("bias"), task_type=config_dict.get("task_type"),
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH, config=config)
    model = model.merge_and_unload()
    model.to(device)
    model.eval()
    print("--- Model and adapter ready on CPU. ---")
    
    # --- 4. INTERACTIVE LOOP ---
    print("\n--- AI GYM COACH IS READY ---")
    while True:
        try:
            user_query = input("\nYour Query: ")
            if user_query.lower() in ["exit", "quit"]: break
            print("--> Retrieving documents...")
            docs = retriever.invoke(user_query)
            context = "\n".join([doc.page_content for doc in docs])
            
            prompt = f"<start_of_turn>user\nYou are an expert AI Gym Coach... Use the context... If not relevant, use your general knowledge.\n\nContext:\n{context}\n\nQuestion: {user_query}<end_of_turn>\n<start_of_turn>model"
            
            print("--> Generating response...")
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            prompt_len = inputs["input_ids"].shape[1]
            
            # Add the generation quality parameters back in
            outputs = model.generate(
                **inputs, 
                max_new_tokens=512, 
                temperature=0.7, 
                repetition_penalty=1.15, 
                no_repeat_ngram_size=4
            )

            # THIS IS THE DEFINITIVE FIX:
            # We decode only the newly generated tokens by slicing the output tensor.
            newly_generated_tokens = outputs[0, prompt_len:]
            final_response = tokenizer.decode(newly_generated_tokens, skip_special_tokens=True)

            print(f"\nAI Gym Coach: {final_response}")
            
        except (KeyboardInterrupt, EOFError):
            print("\nEnding session. Goodbye!")
            break

if __name__ == "__main__":
    main()