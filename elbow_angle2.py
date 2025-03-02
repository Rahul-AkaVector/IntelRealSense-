import cv2
import mediapipe as mp
import pyrealsense2 as rs
import numpy as np
from collections import deque
import math

def calculate_angle(a, b, c):
    """Calculates the angle between three points using the cosine rule."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    ab = a - b
    cb = c - b
    
    dot_product = np.dot(ab, cb)
    magnitude = np.linalg.norm(ab) * np.linalg.norm(cb)
    
    if magnitude == 0:
        return 0
    
    angle = np.arccos(dot_product / magnitude)
    return np.degrees(angle)

# Initialize elbow angle tracking with deque to filter noise
left_elbow_angles = deque(maxlen=10)
right_elbow_angles = deque(maxlen=10)

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
pipeline.start(config)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

try:
    cv2.namedWindow("Elbow Tracking", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Elbow Tracking", 900, 630)  # Set window to 70% of 1280x720 resolution
    
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_result = pose.process(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        h, w, _ = frame.shape
        
        # Estimate elbow angles using Pose landmarks
        if pose_result.pose_landmarks:
            landmarks = pose_result.pose_landmarks.landmark
            left_shoulder = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h)
            left_elbow = (landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x * w, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y * h)
            left_wrist = (landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x * w, landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y * h)
            
            right_shoulder = (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h)
            right_elbow = (landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y * h)
            right_wrist = (landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y * h)
            
            left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
            
            left_elbow_angles.append(left_angle)
            right_elbow_angles.append(right_angle)
            
            filtered_left_angle = np.median(left_elbow_angles)
            filtered_right_angle = np.median(right_elbow_angles)
            
            cv2.putText(frame, f'Left Elbow Angle: {int(filtered_left_angle)}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f'Right Elbow Angle: {int(filtered_right_angle)}', (w - 300, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.line(frame, (int(left_shoulder[0]), int(left_shoulder[1])), (int(left_elbow[0]), int(left_elbow[1])), (255, 0, 0), 2)
            cv2.line(frame, (int(left_elbow[0]), int(left_elbow[1])), (int(left_wrist[0]), int(left_wrist[1])), (255, 0, 0), 2)
            cv2.line(frame, (int(right_shoulder[0]), int(right_shoulder[1])), (int(right_elbow[0]), int(right_elbow[1])), (255, 0, 0), 2)
            cv2.line(frame, (int(right_elbow[0]), int(right_elbow[1])), (int(right_wrist[0]), int(right_wrist[1])), (255, 0, 0), 2)
            
            cv2.circle(frame, (int(left_elbow[0]), int(left_elbow[1])), 5, (0, 0, 255), -1)
            cv2.circle(frame, (int(right_elbow[0]), int(right_elbow[1])), 5, (0, 0, 255), -1)
        
        cv2.imshow("Elbow Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    pass
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
