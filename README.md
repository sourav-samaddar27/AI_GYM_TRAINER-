# AI_GYM_TRAINER-
This application is built using Deep learning(LSTM Autoencoders) and Generative AI( LLM using Gemma-2b)

# AI Personal Trainer

## Project Overview

This repository contains the complete codebase for a sophisticated AI Personal Trainer. The project is divided into two main components:

1.  **Real-Time Form Coach:** A computer vision system that uses a webcam to identify a user's exercise and provide real-time feedback on their form ("CORRECT" or "INCORRECT").
2.  **AI Gym Chatbot:** A conversational AI, powered by a fine-tuned Large Language Model (LLM) and a Retrieval-Augmented Generation (RAG) system, capable of answering a wide range of questions about fitness.



## How It Works

This project is built from several independent AI models that work together. The following steps outline the complete end-to-end workflow, from raw data to a fully functional application.

### Part 1: The Computer Vision Pipeline (The "Eyes")

The Form Coach is a complex system that learns to see and understand exercises.

1.  **Data Collection (`01_collect_raw_data.py`):** The process begins by collecting raw video data of exercises. The script uses **MediaPipe** to analyze these videos and extract the 3D coordinates of 33 body landmarks (the "skeleton") for every frame. This landmark data is much more efficient to train on than raw video. Two types of landmark datasets are created:
    *   **Classifier Data:** From short, augmented clips of exercises (both correct and incorrect form).
    *   **Autoencoder Data:** From longer, full-repetition videos of *only* correct form.

2.  **Model Training (`train_final.py`):** This is the core training script. It builds two types of models:
    *   An **LSTM Classifier** is trained on the first dataset to become an expert at distinguishing between different exercises (e.g., dumbbell curls vs. shoulder press).
    *   Multiple **LSTM Autoencoders** are trained on the second dataset. Each autoencoder becomes a specialist "expert" on the perfect form for a single exercise.

3.  **Threshold Calibration (`03_calibrate_thresholds.py`):** After training, this script runs the autoencoders on correct-form data to determine a "reconstruction error" threshold. Any movement that deviates too far from this threshold is considered "incorrect."

4.  **Real-Time Inference (`realtime_final.py`):** This script ties the CV pipeline together. It uses a webcam, identifies the exercise with the classifier, and then uses the appropriate autoencoder to judge the form against the calibrated threshold, providing live feedback.


### Part 2: The Conversational AI Pipeline (The "Brain")

The Chatbot is a powerful RAG (Retrieval-Augmented Generation) system.

1.  **Data Curation & Cleaning (`clean_dataset_v2.py`):** The process starts with a large collection of fitness documents (PDFs and CSVs). This script performs a deep cleaning, using language detection and vocabulary checks to filter out all non-English, nonsensical, or low-quality text, resulting in a high-quality knowledge base.

2.  **Knowledge Library Creation (`create_rag_database.py`):** This script takes all the clean documents and uses a powerful embedding model (**BAAI/bge-small-en-v1.5**) to convert every piece of text into a mathematical vector. It stores these vectors in a **FAISS** vector database. This database acts as the AI's searchable, long-term memory.

3.  **LLM Fine-Tuning (`fine_tune_gemma.py`):** This is where we create the AI's "personality."
    *   **Download the Base Model:** First, a powerful pre-trained language model, **`google/gemma-2b-it`**, is downloaded. This model already has a vast understanding of language and reasoning.
    *   **Train the Adapter:** The script then uses a technique called **QLoRA** to efficiently fine-tune the base Gemma model on your cleaned fitness data. It doesn't retrain the whole model; instead, it creates a small, highly specialized **LoRA adapter** (`gemma_gym_coach_v3_standard`). This adapter teaches the generic Gemma model how to think, talk, and behave like an expert AI Gym Coach.

4.  **Conversational Inference (`test_rag_llm.py`):** This script combines the components. When a user asks a question:
    *   The RAG system searches the FAISS database to find the most relevant facts.
    *   These facts are passed to the fine-tuned Gemma model (with the adapter loaded).
    *   The model then uses its coaching "skill" and the provided facts to generate a coherent and helpful answer.

---

## How to Run This Project

This project requires a specific setup to function correctly. The recommended approach is to use a cloud environment like **Google Colab** or **Kaggle** for the heavy training and data processing steps, and then run the final application locally.

### Step 1: Download the Pre-Trained Base LLM

To fine-tune or run the chatbot, you first need the base language model.

**Script:** `download_gemma.py`

**Action:** Run this script in a local Python environment. It will download the `google/gemma-2b-it` model (approximately 5 GB) and save it to a `google/gemma-2b-it` folder. This is a one-time download.

# Make sure you are logged into Hugging Face
# huggingface-cli login

python download_gemma.py

Step 2: Fine-Tune the Gemma Model
This step is computationally expensive and must be performed in a cloud environment with a GPU (Kaggle or Google Colab are recommended).

Script: fine_tune_gemma.py

Action:

Upload your cleaned fitness data (e.g., the truly_cleaned_data_v2 folder) to the cloud environment.

Run the fine_tune_gemma.py script. It will use the cloud's GPU to train the gemma_gym_coach_v3_standard adapter.

Download the resulting adapter folder to your local machine.

Step 3: Run the Final Application
Once you have all the components (the base model, the trained adapter, the RAG database, and the CV models), you can run the final application. Due to the project's complexity, this is best achieved using a microservices architecture as outlined in the project's scripts (cv_api.py, rag_api.py, app.py).

End Result
The end result of this entire pipeline is a fully functional AI Personal Trainer that can:

Visually analyze user exercises in real-time and provide immediate corrective feedback.

Conversationally answer a vast range of user questions about fitness, drawing knowledge from a custom-built library and communicating with the personality of an expert coach.
