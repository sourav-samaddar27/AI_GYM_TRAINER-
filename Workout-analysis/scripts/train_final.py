# scripts/train_final.py (DEFINITIVE v13 - DUAL DATA PIPELINE - FULL CODE)

import numpy as np
import os
import json
import time
import pandas as pd
import mediapipe as mp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, auc, precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical
from itertools import cycle

# --- Configuration ---
CLASSIFIER_LANDMARKS_DIR = 'data/landmarks_classifier'
AUTOENCODER_LANDMARKS_DIR = 'data/landmarks_autoencoder'
MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)

SEQUENCE_LENGTH, BATCH_SIZE, EPOCHS = 30, 32, 100
VALIDATION_SPLIT = 0.2
EXERCISES_TO_FIND = ['dumbbel_curls', 'shoulder_press', 'shoulder_side_raises']

mp_pose = mp.solutions.pose
LM = mp_pose.PoseLandmark

# --- HELPER & FEATURE ENGINEERING ---
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180.0 else 360 - angle

def get_coords(landmarks, landmark_enum):
    if landmarks is None or len(landmarks) == 0: return None
    idx = landmark_enum.value * 4
    if idx + 3 < len(landmarks) and landmarks[idx+3] > 0.5:
        return np.array([landmarks[idx], landmarks[idx+1], landmarks[idx+2]])
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
    return all_features

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

# --- VISUALIZATION FUNCTIONS ---
def plot_training_history(history, model_name, plot_accuracy=True):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(history.history['loss'], color='tab:blue', label='Train Loss')
    ax1.plot(history.history['val_loss'], color='tab:orange', label='Validation Loss')
    ax1.set_title(f'{model_name.replace("_", " ").title()} Training History', fontsize=16)
    ax1.set_xlabel('Epoch', fontsize=12); ax1.set_ylabel('Loss', color='tab:blue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    if plot_accuracy and 'accuracy' in history.history:
        ax2 = ax1.twinx()
        ax2.plot(history.history['accuracy'], color='tab:green', linestyle='--', label='Train Accuracy')
        ax2.plot(history.history['val_accuracy'], color='tab:red', linestyle='--', label='Validation Accuracy')
        ax2.set_ylabel('Accuracy', color='tab:red', fontsize=12); ax2.tick_params(axis='y', labelcolor='tab:red')
    fig.tight_layout()
    lines, labels = ax1.get_legend_handles_labels()
    if plot_accuracy and 'accuracy' in history.history:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='best')
    else: ax1.legend(loc='best')
    save_path = os.path.join(MODELS_DIR, f'{model_name}_training_history.png')
    plt.savefig(save_path); plt.close()
    print(f"Training history plot saved to {save_path}")

def plot_performance_metrics(report, class_names):
    try:
        report_data = []
        lines = report.split('\n')
        for line in lines[2:-5]:
            row_data = [val for val in line.split(' ') if val]
            if len(row_data) > 0 and row_data[0] in class_names:
                report_data.append({'class': row_data[0], 'precision': float(row_data[1]), 'recall': float(row_data[2]), 'f1-score': float(row_data[3])})
        df = pd.DataFrame.from_dict(report_data); df.set_index('class', inplace=True)
        df.plot(kind='bar', figsize=(12, 8), colormap='viridis')
        plt.title('Classifier Performance Metrics per Exercise', fontsize=16); plt.ylabel('Score', fontsize=12); plt.xlabel('Exercise', fontsize=12)
        plt.xticks(rotation=45, ha="right"); plt.tight_layout()
        save_path = os.path.join(MODELS_DIR, 'classifier_performance_metrics.png')
        plt.savefig(save_path); plt.close()
        print(f"Performance metrics bar graph saved to {save_path}")
    except (ImportError, IndexError, ValueError) as e:
        print(f"Skipping performance metrics plot due to an error: {e}")

def plot_roc_curves(y_true_cat, y_pred_probs, class_names):
    n_classes = len(class_names)
    fpr, tpr, roc_auc = dict(), dict(), dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_cat[:, i], y_pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_cat.ravel(), y_pred_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes): mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr; tpr["macro"] = mean_tpr; roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    plt.figure(figsize=(12, 10))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'purple'])
    for i, color in zip(range(n_classes), colors): plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC curve of {class_names[i]} (AUC = {roc_auc[i]:.2f})')
    plt.plot(fpr["micro"], tpr["micro"], label=f'Micro-average ROC curve (AUC = {roc_auc["micro"]:.2f})', color='deeppink', linestyle=':', linewidth=4)
    plt.plot(fpr["macro"], tpr["macro"], label=f'Macro-average ROC curve (AUC = {roc_auc["macro"]:.2f})', color='navy', linestyle=':', linewidth=4)
    plt.plot([0, 1], [0, 1], 'k--', lw=2); plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12); plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=16); plt.legend(loc="lower right")
    save_path = os.path.join(MODELS_DIR, 'classifier_roc_auc_curves.png')
    plt.savefig(save_path); plt.close()
    print(f"ROC/AUC curve plot saved to {save_path}")

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    plt.figure(figsize=(10, 8)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 16})
    plt.title('Classifier Confusion Matrix', fontsize=20); plt.ylabel('Actual Class', fontsize=15); plt.xlabel('Predicted Class', fontsize=15)
    plt.xticks(rotation=45); plt.yticks(rotation=0); plt.tight_layout()
    save_path = os.path.join(MODELS_DIR, 'classifier_confusion_matrix.png')
    plt.savefig(save_path); plt.close()
    print(f"Confusion matrix plot saved to {save_path}")

def plot_mae_distribution(mae_correct, mae_incorrect, exercise_name):
    plt.figure(figsize=(10, 6)); sns.kdeplot(mae_correct, fill=True, color="g", label='Correct Form MAE'); sns.kdeplot(mae_incorrect, fill=True, color="r", label='Incorrect Form MAE')
    plt.legend(); plt.title(f'Reconstruction Error Distribution for {exercise_name.replace("_", " ").title()}', fontsize=16)
    plt.xlabel("Mean Absolute Error (MAE)", fontsize=12); plt.ylabel("Density", fontsize=12)
    save_path = os.path.join(MODELS_DIR, f'{exercise_name}_mae_distribution.png')
    plt.savefig(save_path); plt.close()
    print(f"MAE distribution plot saved to {save_path}")
    
def plot_form_precision_recall_curve(mae_correct, mae_incorrect, exercise_name):
    y_true = np.concatenate([np.zeros(len(mae_correct)), np.ones(len(mae_incorrect))])
    y_scores = np.concatenate([mae_correct, mae_incorrect])
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = np.divide(2 * recall * precision, recall + precision, out=np.zeros_like(recall), where=(recall + precision) != 0)
    if len(thresholds) == len(f1_scores): best_threshold_idx = np.argmax(f1_scores)
    else: best_threshold_idx = np.argmax(f1_scores[:-1])
    best_threshold = thresholds[best_threshold_idx]
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='b', label='Precision-Recall curve')
    plt.scatter(recall[best_threshold_idx], precision[best_threshold_idx], marker='o', color='red', label=f'Best Balanced Threshold ({best_threshold:.3f})', s=100, zorder=5)
    plt.xlabel('Recall (Sensitivity)', fontsize=12); plt.ylabel('Precision', fontsize=12)
    plt.title(f'Form Correction Precision-Recall Curve for {exercise_name.replace("_", " ").title()}', fontsize=16)
    plt.legend(); plt.grid(True)
    save_path = os.path.join(MODELS_DIR, f'{exercise_name}_form_precision_recall_curve.png')
    plt.savefig(save_path); plt.close()
    print(f"Form P-R curve saved to {save_path}")

def train_all_models():
    # --- PART 1: TRAIN CLASSIFIER ---
    print("\n" + "="*50 + "\nPART 1: TRAINING EXERCISE CLASSIFIER\n" + "="*50)
    
    # Use the massive augmented dataset for the classifier
    RAW_LANDMARKS_DIR = CLASSIFIER_LANDMARKS_DIR
    
    exercises_with_data = [ex for ex in EXERCISES_TO_FIND if os.path.exists(os.path.join(RAW_LANDMARKS_DIR, ex)) and len(os.listdir(os.path.join(RAW_LANDMARKS_DIR, ex))) > 0]
    if len(exercises_with_data) < 2: print("FATAL: Need at least two exercises with data for the classifier."); return
    print(f"Found classifier data for: {exercises_with_data}")
    EXERCISES = exercises_with_data; NUM_CLASSES = len(EXERCISES)
    EXERCISE_CLASS_MAP = {name: i for i, name in enumerate(EXERCISES)}
    X, y = [], []
    for exercise, label in EXERCISE_CLASS_MAP.items():
        for posture in ['correct', 'incorrect']:
            posture_dir = os.path.join(RAW_LANDMARKS_DIR, exercise, posture)
            if not os.path.exists(posture_dir): continue
            for npy_file in os.listdir(posture_dir):
                seq = np.load(os.path.join(posture_dir, npy_file))
                features = get_classifier_features(seq)
                step = int(SEQUENCE_LENGTH * 0.5)
                for i in range(0, len(features) - SEQUENCE_LENGTH + 1, max(1, step)): X.append(features[i:i+SEQUENCE_LENGTH]); y.append(label)
    
    if not X: print("FATAL: No data for classifier training."); return
    X, y = np.array(X), np.array(y)
    num_samples, _, num_features = X.shape
    print(f"Loaded {num_samples} sequences with {num_features} features for classifier.")
    
    class_labels = np.unique(y)
    class_weights = compute_class_weight(class_weight='balanced', classes=class_labels, y=y)
    class_weight_dict = dict(zip(class_labels, class_weights))
    print("\n--- Calculated Class Weights ---")
    for i, exercise in enumerate(EXERCISES): print(f"'{exercise}': {class_weight_dict.get(i, 1.0):.2f}")
    print("----------------------------\n")

    scaler = StandardScaler().fit(X.reshape(-1, num_features))
    X_scaled = scaler.transform(X.reshape(-1, num_features)).reshape(X.shape)
    y_cat = to_categorical(y, num_classes=NUM_CLASSES)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_cat, test_size=VALIDATION_SPLIT, random_state=42, stratify=y)
    
    model = Sequential([ LSTM(64, activation='tanh', return_sequences=True, input_shape=(SEQUENCE_LENGTH, num_features)), Dropout(0.4), LSTM(32, activation='tanh'), Dropout(0.4), Dense(32, activation='relu'), Dense(NUM_CLASSES, activation='softmax')])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    ckpt = ModelCheckpoint(os.path.join(MODELS_DIR, 'exercise_classifier_best_model.h5'), save_best_only=True, monitor='val_accuracy', mode='max')
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    history_classifier = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test), callbacks=[ckpt, early_stop], class_weight=class_weight_dict, verbose=1)
    
    with open(os.path.join(MODELS_DIR, 'exercise_classifier_scaler_params.json'), 'w') as f: json.dump({'mean': scaler.mean_.tolist(), 'std': scaler.scale_.tolist(), 'class_map': EXERCISE_CLASS_MAP}, f)
    print("--- Saved Classifier Model and Scaler ---")
    
    print("\n--- Evaluating Classifier Performance ---")
    y_pred_probs = model.predict(X_test); y_pred_labels = np.argmax(y_pred_probs, axis=1)
    y_true_labels = np.argmax(y_test, axis=1)
    report = classification_report(y_true_labels, y_pred_labels, target_names=EXERCISES, zero_division=0)
    with open(os.path.join(MODELS_DIR, 'classifier_report.txt'), 'w') as f: f.write(report)
    print("Classification Report:\n", report)
    plot_training_history(history_classifier, "classifier", plot_accuracy=True)
    plot_performance_metrics(report, EXERCISES)
    plot_roc_curves(y_test, y_pred_probs, EXERCISES)
    plot_confusion_matrix(y_true_labels, y_pred_labels, EXERCISES)
    
    # --- PART 2: TRAIN AUTOENCODERS ---
    print("\n" + "="*50 + "\nPART 2: TRAINING FORM-CHECKING AUTOENCODERS\n" + "="*50)
    
    autoencoder_report = ""
    for exercise_name in EXERCISES:
        print(f"\n--- Training & Evaluating Autoencoder for: {exercise_name.upper()} ---")
        
        X_correct_train = []
        correct_dir = os.path.join(AUTOENCODER_LANDMARKS_DIR, exercise_name, 'correct')
        if not os.path.exists(correct_dir): print(f"  WARNING: 'correct' data dir for autoencoder not found. Skipping."); continue
        
        for npy_file in os.listdir(correct_dir):
            seq = np.load(os.path.join(correct_dir, npy_file))
            features = get_form_features(seq)
            step = int(SEQUENCE_LENGTH * 0.5)
            for i in range(0, len(features) - SEQUENCE_LENGTH + 1, max(1, step)): X_correct_train.append(features[i:i+SEQUENCE_LENGTH])
        
        if not X_correct_train: print(f"  WARNING: No 'correct' data loaded for autoencoder. Skipping."); continue
        
        X_correct_train = np.array(X_correct_train)
        num_samples, _, num_features = X_correct_train.shape
        print(f"Loaded {num_samples} 'correct' sequences for {exercise_name} autoencoder training.")
        
        scaler = StandardScaler().fit(X_correct_train.reshape(-1, num_features))
        X_scaled = scaler.transform(X_correct_train.reshape(-1, num_features)).reshape(X_correct_train.shape)
        
        inputs = Input(shape=(SEQUENCE_LENGTH, num_features))
        encoder = LSTM(8, activation='tanh')(inputs)
        decoder = RepeatVector(SEQUENCE_LENGTH)(encoder)
        decoder = LSTM(8, activation='tanh', return_sequences=True)(decoder)
        decoder = Dropout(0.2)(decoder)
        output = TimeDistributed(Dense(num_features))(decoder)
        autoencoder = Model(inputs, output)
        autoencoder.compile(optimizer='adam', loss='mae')
        
        ckpt_ae = ModelCheckpoint(os.path.join(MODELS_DIR, f'{exercise_name}_autoencoder.h5'), save_best_only=True, monitor='val_loss', mode='min')
        early_stop_ae = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        history_autoencoder = autoencoder.fit(X_scaled, X_scaled, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT, callbacks=[ckpt_ae, early_stop_ae], verbose=1)
        
        with open(os.path.join(MODELS_DIR, f'{exercise_name}_autoencoder_scaler.json'), 'w') as f: json.dump({'mean': scaler.mean_.tolist(), 'std': scaler.scale_.tolist()}, f)
        print(f"--- Saved Autoencoder Model and Scaler for {exercise_name.upper()} ---")
        
        print(f"\n--- Evaluating Autoencoder for: {exercise_name.upper()} ---")
        X_correct_eval, X_incorrect_eval = [], []
        correct_eval_dir = os.path.join(AUTOENCODER_LANDMARKS_DIR, exercise_name, 'correct')
        incorrect_eval_dir = os.path.join(CLASSIFIER_LANDMARKS_DIR, exercise_name, 'incorrect')

        if os.path.exists(correct_eval_dir):
            for npy_file in os.listdir(correct_eval_dir):
                seq = np.load(os.path.join(correct_eval_dir, npy_file))
                features = get_form_features(seq)
                step = int(SEQUENCE_LENGTH * 0.5)
                for i in range(0, len(features) - SEQUENCE_LENGTH + 1, max(1, step)): X_correct_eval.append(features[i:i+SEQUENCE_LENGTH])
        if os.path.exists(incorrect_eval_dir):
            for npy_file in os.listdir(incorrect_eval_dir):
                seq = np.load(os.path.join(incorrect_eval_dir, npy_file))
                features = get_form_features(seq)
                step = int(SEQUENCE_LENGTH * 0.5)
                for i in range(0, len(features) - SEQUENCE_LENGTH + 1, max(1, step)): X_incorrect_eval.append(features[i:i+SEQUENCE_LENGTH])

        if X_correct_eval and X_incorrect_eval:
            X_correct_eval, X_incorrect_eval = np.array(X_correct_eval), np.array(X_incorrect_eval)
            X_correct_scaled_eval = scaler.transform(X_correct_eval.reshape(-1, num_features)).reshape(X_correct_eval.shape)
            reconstructions_correct = autoencoder.predict(X_correct_scaled_eval)
            mae_correct = np.mean(np.abs(reconstructions_correct - X_correct_scaled_eval), axis=(1, 2))
            
            X_incorrect_scaled_eval = scaler.transform(X_incorrect_eval.reshape(-1, num_features)).reshape(X_incorrect_eval.shape)
            reconstructions_incorrect = autoencoder.predict(X_incorrect_scaled_eval)
            mae_incorrect = np.mean(np.abs(reconstructions_incorrect - X_incorrect_scaled_eval), axis=(1, 2))
            
            plot_mae_distribution(mae_correct, mae_incorrect, exercise_name)
            plot_training_history(history_autoencoder, f'{exercise_name}_autoencoder', plot_accuracy=False)
            plot_form_precision_recall_curve(mae_correct, mae_incorrect, exercise_name)
            report_line = f"\n----- {exercise_name.upper()} -----\nMean MAE on CORRECT form:   {np.mean(mae_correct):.5f}\nMean MAE on INCORRECT form: {np.mean(mae_incorrect):.5f}"
            autoencoder_report += report_line
            print(report_line)
        else:
            print("Not enough correct/incorrect data to generate evaluation plots.")
            plot_training_history(history_autoencoder, f'{exercise_name}_autoencoder', plot_accuracy=False)

    with open(os.path.join(MODELS_DIR, 'autoencoder_evaluation_report.txt'), 'w') as f: f.write(autoencoder_report)
    print(f"\nAutoencoder evaluation report saved to {os.path.join(MODELS_DIR, 'autoencoder_evaluation_report.txt')}")

if __name__ == '__main__':
    train_all_models()