# scripts/realtime_final.py (DEFINITIVE v13 - DUAL DATA PIPELINE)
import cv2
import mediapipe as mp
import numpy as np
import json
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from collections import deque
import os

# --- ROBUST PATH CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)); PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
# ----------------------------------

SEQUENCE_LENGTH = 30; CONFIDENCE_THRESHOLD = 0.9

SENSITIVITY_SETTINGS = {
    'dumbbel_curls': 1.5,
    'shoulder_press': 1.5,
    'shoulder_side_raises': 1.5,
    'default': 1.0
}
# -----------------------------

mp_pose = mp.solutions.pose; LM = mp_pose.PoseLandmark; mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- Feature Extractors (MUST BE 100% IDENTICAL TO train_final.py) ---
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

def get_classifier_features(sequence):
    all_features = []
    for i in range(len(sequence)):
        lm = sequence[i]
        L_SHOULDER, R_SHOULDER = get_coords(lm, LM.LEFT_SHOULDER), get_coords(lm, LM.RIGHT_SHOULDER)
        if L_SHOULDER is None or R_SHOULDER is None: all_features.append(np.zeros(22)); continue
        shoulder_midpoint = (L_SHOULDER + R_SHOULDER) / 2
        shoulder_dist = np.linalg.norm(L_SHOULDER - R_SHOULDER)
        if shoulder_dist < 0.05: shoulder_dist = 0.05
        L_ELBOW, R_ELBOW, L_WRIST, R_WRIST, L_HIP, R_HIP = get_coords(lm, LM.LEFT_ELBOW), get_coords(lm, LM.RIGHT_ELBOW), get_coords(lm, LM.LEFT_WRIST), get_coords(lm, LM.RIGHT_WRIST), get_coords(lm, LM.LEFT_HIP), get_coords(lm, LM.RIGHT_HIP)
        l_elbow_pos = (L_ELBOW - shoulder_midpoint) / shoulder_dist if L_ELBOW is not None else np.zeros(3)
        r_elbow_pos = (R_ELBOW - shoulder_midpoint) / shoulder_dist if R_ELBOW is not None else np.zeros(3)
        l_wrist_pos = (L_WRIST - shoulder_midpoint) / shoulder_dist if L_WRIST is not None else np.zeros(3)
        r_wrist_pos = (R_WRIST - shoulder_midpoint) / shoulder_dist if R_WRIST is not None else np.zeros(3)
        l_wrist_vel, r_wrist_vel = np.zeros(3), np.zeros(3)
        if i > 0:
            prev_lm = sequence[i-1]
            PREV_L_WRIST, PREV_R_WRIST = get_coords(prev_lm, LM.LEFT_WRIST), get_coords(prev_lm, LM.RIGHT_WRIST)
            if L_WRIST is not None and PREV_L_WRIST is not None: l_wrist_vel = (L_WRIST - PREV_L_WRIST) / shoulder_dist
            if R_WRIST is not None and PREV_R_WRIST is not None: r_wrist_vel = (R_WRIST - PREV_R_WRIST) / shoulder_dist
        l_elbow_angle = calculate_angle(L_SHOULDER, L_ELBOW, L_WRIST) / 180.0 if all(v is not None for v in [L_SHOULDER, L_ELBOW, L_WRIST]) else 0
        r_elbow_angle = calculate_angle(R_SHOULDER, R_ELBOW, R_WRIST) / 180.0 if all(v is not None for v in [R_SHOULDER, R_ELBOW, R_WRIST]) else 0
        l_shoulder_angle = calculate_angle(L_HIP, L_SHOULDER, L_ELBOW) / 180.0 if all(v is not None for v in [L_HIP, L_SHOULDER, L_ELBOW]) else 0
        r_shoulder_angle = calculate_angle(R_HIP, R_SHOULDER, R_ELBOW) / 180.0 if all(v is not None for v in [R_HIP, R_SHOULDER, R_ELBOW]) else 0
        l_wrist_shoulder_y_dist = (L_SHOULDER[1] - L_WRIST[1]) if L_WRIST is not None else 0
        r_wrist_shoulder_y_dist = (R_SHOULDER[1] - R_WRIST[1]) if R_WRIST is not None else 0
        frame_features = np.concatenate([l_elbow_pos, r_elbow_pos, l_wrist_pos, r_wrist_pos, l_wrist_vel, r_wrist_vel, [l_elbow_angle, r_elbow_angle, l_shoulder_angle, r_shoulder_angle], [l_wrist_shoulder_y_dist, r_wrist_shoulder_y_dist]]).flatten()
        all_features.append(frame_features)
    return np.array(all_features)

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
    return np.array(all_features)

def main():
    try:
        print(f"Loading models from: {MODELS_DIR}")
        classifier_model = load_model(os.path.join(MODELS_DIR, 'exercise_classifier_best_model.h5'))
        with open(os.path.join(MODELS_DIR, 'exercise_classifier_scaler_params.json'), 'r') as f: params = json.load(f)
        classifier_scaler = StandardScaler(); classifier_scaler.mean_ = np.array(params['mean']); classifier_scaler.scale_ = np.array(params['std'])
        REVERSE_EXERCISE_MAP = {v: k for k, v in params['class_map'].items()}
        with open(os.path.join(MODELS_DIR, 'mae_thresholds.json'), 'r') as f: THRESHOLDS = json.load(f)
        autoencoder_models = {}
        for exercise_name in REVERSE_EXERCISE_MAP.values():
            model = load_model(os.path.join(MODELS_DIR, f'{exercise_name}_autoencoder.h5'), compile=False)
            with open(os.path.join(MODELS_DIR, f'{exercise_name}_autoencoder_scaler.json'), 'r') as f: scaler_params = json.load(f)
            scaler = StandardScaler(); scaler.mean_ = np.array(scaler_params['mean']); scaler.scale_ = np.array(scaler_params['std'])
            autoencoder_models[exercise_name] = {'model': model, 'scaler': scaler}
        print("All models and thresholds loaded successfully.")
    except Exception as e: print(f"FATAL: Could not load models. Error: {e}"); exit()
    
    current_mode = 'IDENTIFYING'; identified_exercise = "NONE"
    landmarks_buffer = deque(maxlen=SEQUENCE_LENGTH)
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read();
        if not ret: break
        
        frame = cv2.flip(frame, 1); image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image); image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        status_text = ""; prediction_confidence = 0.0
        can_process_form = False

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks_flat = [lm for lm_obj in results.pose_landmarks.landmark for lm in [lm_obj.x, lm_obj.y, lm_obj.z, lm_obj.visibility]]
            landmarks_buffer.append(landmarks_flat)
            
            last_frame = landmarks_buffer[-1]
            l_shoulder, l_elbow, l_wrist = get_coords(last_frame, LM.LEFT_SHOULDER), get_coords(last_frame, LM.LEFT_ELBOW), get_coords(last_frame, LM.LEFT_WRIST)
            r_shoulder, r_elbow, r_wrist = get_coords(last_frame, LM.RIGHT_SHOULDER), get_coords(last_frame, LM.RIGHT_ELBOW), get_coords(last_frame, LM.RIGHT_WRIST)
            if (l_shoulder is not None and l_elbow is not None and l_wrist is not None) or \
               (r_shoulder is not None and r_elbow is not None and r_wrist is not None):
                can_process_form = True

            if len(landmarks_buffer) == SEQUENCE_LENGTH:
                if current_mode == 'IDENTIFYING':
                    if can_process_form:
                        features = get_classifier_features(landmarks_buffer)
                        input_scaled = classifier_scaler.transform(features).reshape(1, SEQUENCE_LENGTH, -1)
                        prediction = classifier_model.predict(input_scaled, verbose=0)[0]
                        pred_idx, prediction_confidence = np.argmax(prediction), np.max(prediction)
                        if prediction_confidence > CONFIDENCE_THRESHOLD:
                            exercise_name = REVERSE_EXERCISE_MAP.get(pred_idx)
                            if exercise_name and exercise_name in autoencoder_models:
                                identified_exercise = exercise_name; current_mode = 'ANALYZING'
                                print(f"Switched to ANALYZING for {identified_exercise.upper()}")
                                landmarks_buffer.clear()
                    if not can_process_form and identified_exercise == "NONE":
                        status_text = "REPOSITION"
                    else:
                        status_text = "IDENTIFYING..."
                
                elif current_mode == 'ANALYZING':
                    if can_process_form:
                        features = get_form_features(landmarks_buffer)
                        autoencoder_info = autoencoder_models[identified_exercise]
                        input_scaled = autoencoder_info['scaler'].transform(features).reshape(1, SEQUENCE_LENGTH, -1)
                        reconstruction = autoencoder_info['model'].predict(input_scaled, verbose=0)
                        mae = np.mean(np.abs(reconstruction - input_scaled))
                        prediction_confidence = mae
                        sensitivity = SENSITIVITY_SETTINGS.get(identified_exercise, SENSITIVITY_SETTINGS['default'])
                        current_threshold = THRESHOLDS.get(identified_exercise, float('inf')) * sensitivity
                        if mae < current_threshold: status_text = "CORRECT"
                        else: status_text = "INCORRECT"
                    else:
                        status_text = "REPOSITION"
        
        # --- UI Display ---
        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, 'EXERCISE:', (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, identified_exercise.replace("_", " ").upper(), (160, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.rectangle(image, (0, image.shape[0] - 50), (640, image.shape[0]), (100, 100, 100), -1)
        if status_text == "REPOSITION": status_color = (0, 255, 255)
        elif "CORRECT" in status_text: status_color = (0, 255, 0)
        elif current_mode == 'IDENTIFYING': status_color = (255, 255, 0)
        else: status_color = (0, 0, 255)
        cv2.putText(image, 'STATUS:', (15, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, status_text, (130, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2, cv2.LINE_AA)
        display_value_text = f'{prediction_confidence:.4f}' if current_mode == 'ANALYZING' else f'{prediction_confidence:.2f}'
        cv2.putText(image, display_value_text, (480, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, "Press 'r' to reset", (15, image.shape[0] - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow('AI Gym Guide', image)
        
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'): break
        if key == ord('r'):
            current_mode, identified_exercise = 'IDENTIFYING', "NONE"; landmarks_buffer.clear()
            print("\n--- RESET ---\n")
    cap.release(); cv2.destroyAllWindows()

if __name__ == '__main__':
    main()