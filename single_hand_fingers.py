import cv2
import mediapipe as mp
import pyrealsense2 as rs
import numpy as np

def is_finger_open(hand_landmarks, finger_indices):
    """Determines if a finger is open or closed based on landmark positions."""
    tip = hand_landmarks.landmark[finger_indices[0]]
    pip = hand_landmarks.landmark[finger_indices[1]]
    return tip.y < pip.y  # If tip is above PIP joint, it's open

def get_finger_status(hand_landmarks):
    """Returns a dictionary indicating which fingers are open or closed."""
    fingers = {1: [4, 3], 2: [8, 6], 3: [12, 10], 4: [16, 14], 5: [20, 18]}
    status = {"Open": [], "Closed": []}
    
    for finger, indices in fingers.items():
        if is_finger_open(hand_landmarks, indices):
            status["Open"].append(finger)
        else:
            status["Closed"].append(finger)
    
    return status

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Extract coordinates
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    h, w, _ = frame.shape
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv2.putText(frame, f'{idx}', (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    
                # Get finger open/close status
                status = get_finger_status(hand_landmarks)
                text = f'Closed: {status["Closed"]}, Open: {status["Open"]}'
                cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        cv2.imshow("Hand Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    pass
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
