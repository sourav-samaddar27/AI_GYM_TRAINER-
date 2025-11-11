# scripts/03_calibrate_thresholds.py (DEFINITIVE v13 - DUAL DATA PIPELINE)

import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import json
import mediapipe as mp
import time

# --- DYNAMIC PATH CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)); PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
# We calibrate using the clean, full-rep landmark data
RAW_LANDMARKS_DIR = os.path.join(PROJECT_ROOT, 'data', 'landmarks_autoencoder')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
# ----------------------------------

SEQUENCE_LENGTH = 30; PERCENTILE = 98 
mp_pose = mp.solutions.pose; LM = mp_pose.PoseLandmark

# --- Feature Extractors (MUST BE 100% IDENTICAL TO train_final.py's form features) ---
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180.0 else 360 - angle

def get_coords(landmarks, landmark_enum):
    if landmarks is None or len(landmarks) == 0: return None
    idx = landmark_enum.value * 4
    if idx + 3 < len(landmarks) and landmarks[idx+3] > 0.5: return np.array([landmarks[idx], landmarks[idx+1], landmarks[idx+2]])
    return None

def get_form_features(sequence):
    all_features = []
    for i in range(len(sequence)):
        lm = sequence[i]
        L_SHOULDER, R_SHOULDER = get_coords(lm, LM.LEFT_SHOULDER), get_coords(lm, LM.RIGHT_SHOULDER)
        if L_SHOULDER is None or R_SHOULDER is None: all_features.append(np.zeros(14)); continue
        shoulder_midpoint = (L_SHOULDER + R_SHOULDER) / 2
        shoulder_dist = np.linalg.norm(L_SHOULDER - R_SHOULDER)
        if shoulder_dist < 0.05: shoulder_dist = 0.05
        L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = get_coords(lm, LM.LEFT_ELBOW), get_coords(lm, LM.RIGHT_ELBOW), get_coords(lm, LM.LEFT_WRIST), get_coords(lm, LM.RIGHT_WRIST)
        l_elbow_pos = (L_ELBOW - shoulder_midpoint) / shoulder_dist if L_ELBOW is not None else np.zeros(3)
        r_elbow_pos = (R_ELBOW - shoulder_midpoint) / shoulder_dist if R_ELBOW is not None else np.zeros(3)
        l_wrist_pos = (L_WRIST - shoulder_midpoint) / shoulder_dist if L_WRIST is not None else np.zeros(3)
        r_wrist_pos = (R_WRIST - shoulder_midpoint) / shoulder_dist if R_WRIST is not None else np.zeros(3)
        l_elbow_angle = calculate_angle(L_SHOULDER, L_ELBOW, L_WRIST) / 180.0 if all(v is not None for v in [L_SHOULDER, L_ELBOW, L_WRIST]) else 0
        r_elbow_angle = calculate_angle(R_SHOULDER, R_ELBOW, R_WRIST) / 180.0 if all(v is not None for v in [R_SHOULDER, R_ELBOW, R_WRIST]) else 0
        frame_features = np.concatenate([l_elbow_pos, r_elbow_pos, l_wrist_pos, r_wrist_pos, [l_elbow_angle, r_elbow_angle]]).flatten()
        all_features.append(frame_features)
    return all_features

def main():
    start_time = time.time(); thresholds = {}
    print("--- Calculating Optimal Reconstruction Error Thresholds ---")
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    try:
        with open(os.path.join(MODELS_DIR, 'exercise_classifier_scaler_params.json'), 'r') as f:
            params = json.load(f)
        available_exercises = list(params['class_map'].keys())
        print(f"Found trained models for: {available_exercises}")
    except FileNotFoundError:
        print("FATAL: Classifier params not found. Please run train_final.py first.")
        return

    for exercise_name in available_exercises:
        print(f"\nCalibrating for: {exercise_name.upper()}")
        try:
            model = load_model(os.path.join(MODELS_DIR, f'{exercise_name}_autoencoder.h5'), compile=False)
            with open(os.path.join(MODELS_DIR, f'{exercise_name}_autoencoder_scaler.json'), 'r') as f: params = json.load(f)
            scaler = StandardScaler(); scaler.mean_ = np.array(params['mean']); scaler.scale_ = np.array(params['std'])
        except Exception as e: print(f"  Could not load model/scaler for {exercise_name}. Skipping. Error: {e}"); continue
        
        X_correct = []
        correct_dir = os.path.join(RAW_LANDMARKS_DIR, exercise_name, 'correct')
        if not os.path.exists(correct_dir): print(f"  'correct' data directory not found for calibration. Skipping."); continue
        
        for npy_file in os.listdir(correct_dir):
            seq = np.load(os.path.join(correct_dir, npy_file))
            features = get_form_features(seq)
            step = int(SEQUENCE_LENGTH * 0.5)
            for i in range(0, len(features) - SEQUENCE_LENGTH + 1, max(1, step)): X_correct.append(features[i:i+SEQUENCE_LENGTH])
        
        if not X_correct: print(f"  No correct data found for calibration. Skipping."); continue
        
        X_correct = np.array(X_correct)
        X_scaled = scaler.transform(X_correct.reshape(-1, X_correct.shape[2])).reshape(X_correct.shape)
        reconstructions = model.predict(X_scaled, verbose=0)
        mae_losses = np.mean(np.abs(reconstructions - X_scaled), axis=(1, 2))
        calculated_threshold = np.percentile(mae_losses, PERCENTILE)
        thresholds[exercise_name] = calculated_threshold
        print(f"  {PERCENTILE}th Percentile of MAE on 'correct' data: {calculated_threshold:.5f}")
        print(f"  RECOMMENDED THRESHOLD SET TO: {calculated_threshold:.5f}")

    output_path = os.path.join(MODELS_DIR, 'mae_thresholds.json')
    print(f"\nAttempting to save thresholds to: {os.path.abspath(output_path)}")
    try:
        with open(output_path, 'w') as f: json.dump(thresholds, f, indent=4)
        print("File saved successfully!")
    except Exception as e: print(f"ERROR: Failed to save the file. Reason: {e}")
    end_time = time.time()
    print("\n--- CALIBRATION COMPLETE ---")
    print(f"Thresholds have been calculated in {end_time - start_time:.2f} seconds.")
    print(f"Values saved to '{output_path}'")

if __name__ == '__main__':
    main()