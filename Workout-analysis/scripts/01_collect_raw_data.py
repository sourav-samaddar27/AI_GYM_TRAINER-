# scripts/01_collect_raw_data.py (DEFINITIVE v16 - TARGETED & EFFICIENT)

import cv2
import mediapipe as mp
import numpy as np
import os
import time

# --- Configuration ---
CLASSIFIER_VIDEO_DIR = 'data/raw_videos'
AUTOENCODER_VIDEO_DIR = 'data/raw_videos_full_reps'
CLASSIFIER_LANDMARKS_DIR = 'data/landmarks_classifier'
AUTOENCODER_LANDMARKS_DIR = 'data/landmarks_autoencoder'
VALID_EXERCISES = ['dumbbel_curls', 'shoulder_press', 'shoulder_side_raises']

mp_pose = mp.solutions.pose
pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def process_video(video_path, output_dir, exercise_name, posture_type, should_rotate):
    file_name = os.path.splitext(os.path.basename(video_path))[0]
    save_dir = os.path.join(output_dir, exercise_name, posture_type)
    save_path = os.path.join(save_dir, f'{file_name}.npy')
    
    if os.path.exists(save_path):
        return True # Return True to count it as "processed"

    print(f"Processing for {os.path.basename(output_dir)}: [{exercise_name}][{posture_type}] > {os.path.basename(video_path)}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ERROR: Could not open video file: {video_path}. Skipping.")
        return False

    video_landmarks = []
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    is_vertical = height > width
    
    if should_rotate and is_vertical:
        print(f"  -> Vertical video detected. Rotating frames to be upright.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        if should_rotate and is_vertical:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        
        if results.pose_landmarks:
            frame_landmarks = [lm for lm_obj in results.pose_landmarks.landmark for lm in [lm_obj.x, lm_obj.y, lm_obj.z, lm_obj.visibility]]
            video_landmarks.append(frame_landmarks)
            
    cap.release()
    
    if not video_landmarks:
        print(f"  WARNING: No landmarks detected in {video_path}. Skipping.")
        return False

    os.makedirs(save_dir, exist_ok=True) 
    np.save(save_path, np.array(video_landmarks, dtype=np.float32))
    print(f"  SUCCESS: Saved {len(video_landmarks)} frames to {save_path}")
    return True

def main():
    start_time = time.time()
    classifier_count = 0
    autoencoder_count = 0

    # print("\n" + "="*50)
    # print("--- PIPELINE 1: Verifying Classifier Landmark Data (No Rotation) ---")
    # print("="*50)
    # if not os.path.isdir(CLASSIFIER_VIDEO_DIR):
    #     print(f"FATAL ERROR: Classifier video directory not found at '{CLASSIFIER_VIDEO_DIR}'.")
    # else:
    #     for exercise_name in VALID_EXERCISES:
    #         for posture_type in ['correct', 'incorrect']:
    #             video_dir = os.path.join(CLASSIFIER_VIDEO_DIR, exercise_name, posture_type)
    #             if not os.path.isdir(video_dir): continue
                
    #             video_files = [f for f in os.listdir(video_dir) if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))]
    #             for file in video_files:
    #                 if process_video(os.path.join(video_dir, file), CLASSIFIER_LANDMARKS_DIR, exercise_name, posture_type, should_rotate=False):
    #                     classifier_count += 1
    
    print("\n" + "="*50)
    print("--- PIPELINE 2: Processing Autoencoder Landmark Data (With Rotation) ---")
    print("="*50)
    if not os.path.isdir(AUTOENCODER_VIDEO_DIR):
        print(f"WARNING: Full-rep video directory not found at '{AUTOENCODER_VIDEO_DIR}'.")
    else:
        for exercise_name in VALID_EXERCISES:
            posture_type = 'correct'
            video_dir = os.path.join(AUTOENCODER_VIDEO_DIR, exercise_name, posture_type)
            if not os.path.isdir(video_dir): continue

            video_files = [f for f in os.listdir(video_dir) if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))]
            for file in video_files:
                if process_video(os.path.join(video_dir, file), AUTOENCODER_LANDMARKS_DIR, exercise_name, posture_type, should_rotate=True):
                    autoencoder_count += 1
    
    end_time = time.time()
    total_videos = classifier_count + autoencoder_count
    print("\n" + "="*50)
    print("--- Data Collection Complete ---")
    if total_videos == 0:
        print("\nWARNING: No videos were processed or found.")
    else:
        print(f"Verified {classifier_count} classifier videos and processed {autoencoder_count} autoencoder videos in {(end_time - start_time)/60:.2f} minutes.")
        print(f"Classifier landmarks are in '{CLASSIFIER_LANDMARKS_DIR}'")
        print(f"Autoencoder landmarks are in '{AUTOENCODER_LANDMARKS_DIR}'")
    print("="*50)

if __name__ == '__main__':
    main()