import cv2
import mediapipe as mp
import os
import time

# --- SETUP ---
mp_drawing = mp.solutions.drawing_utils
mphands = mp.solutions.hands

# Updated: Specify the exact save directory provided by the user
SAVE_DIR = r"C:\Users\USER\Web development\Meme Hand Tracking\img\67"

# Create the directory if it doesn't exist
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# Initialize MediaPipe Hands with max_num_hands=2
hands = mphands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)
padding = 40  # Padding around the hand for cropping

print("🚀 Multi-Hand Cropper Active")
print(f"📁 Saving crops to: {os.path.abspath(SAVE_DIR)}")
print("Press 'S' to save current frame crops, 'Q' to quit.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Flip the image horizontally for a later selfie-view display
    # Convert the BGR image to RGB.
    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image and find hands
    results = hands.process(rgb_image)
    
    h, w, _ = image.shape
    save_triggered = False  # Flag to track if 'S' was pressed
    
    if results.multi_hand_landmarks:
        # FIXED: Use multi_handedness instead of multi_hand_labels
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Get the label (Left or Right)
            hand_label = handedness.classification[0].label
            
            # Get bounding box coordinates
            x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
            y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]
            
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            # Apply padding and constrain to image boundaries
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)
            
            # Crop the hand
            hand_crop = image[y_min:y_max, x_min:x_max]
            
            # Draw on the main display
            color = (0, 255, 0) if hand_label == "Right" else (255, 0, 0)
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(image, hand_label, (x_min, y_min - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Display the individual crops in separate windows
            if hand_crop.size > 0:
                cv2.imshow(f"{hand_label} Hand Crop", hand_crop)
            
            # Draw landmarks
            mp_drawing.draw_landmarks(image, hand_landmarks, mphands.HAND_CONNECTIONS)

    # Check for keys ONCE per frame
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        # Save ALL detected hands when 'S' is pressed
        if results.multi_hand_landmarks:
            timestamp = int(time.time() * 1000)
            # FIXED: Use multi_handedness here too
            for i, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
                hand_label = handedness.classification[0].label
                
                # Recalculate bounding box for saving
                x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
                y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(w, x_max + padding)
                y_max = min(h, y_max + padding)
                
                hand_crop = image[y_min:y_max, x_min:x_max]
                if hand_crop.size > 0:
                    filename = os.path.join(SAVE_DIR, f"{hand_label}_{timestamp}_{i}.jpg")
                    cv2.imwrite(filename, hand_crop)
                    print(f"✅ Saved: {filename}")
        else:
            print("❌ No hands detected to save!")
    
    elif key == ord('q'):
        break

    cv2.imshow('Multi-Hand Tracker & Cropper', image)

cap.release()
cv2.destroyAllWindows()