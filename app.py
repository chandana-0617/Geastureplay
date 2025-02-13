import cv2
import mediapipe as mp
import pygame
from pygame import mixer
import math

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Initialize Pygame Mixer for Music
mixer.init()

# Load a valid MP3 file (ensure the path and file are correct)
song_path = r'D:\Geastureplay\your_song.mp3'  # Replace with your actual file path
try:
    mixer.music.load(song_path)
except pygame.error as e:
    print(f"Error loading music: {e}. Please check the MP3 file.")
    exit()  # Exit the program if the file is invalid

# Variables to track the gesture state
is_playing = False

# Start Webcam
cap = cv2.VideoCapture(0)

def calculate_distance(p1, p2):
    """Function to calculate the Euclidean distance between two points"""
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame for natural interaction
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame for hand landmarks
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            try:
                # Access fingertips
                thumb_tip = hand_landmarks.landmark[4]  # Thumb Tip (index 4)
                index_tip = hand_landmarks.landmark[8]  # Index Finger Tip (index 8)
                pinky_tip = hand_landmarks.landmark[20]  # Pinky Tip (index 20)
                middle_tip = hand_landmarks.landmark[12]  # Middle Finger Tip (index 12)
                ring_tip = hand_landmarks.landmark[16]  # Ring Finger Tip (index 16)
            except IndexError:
                # If landmarks are not found, skip the current iteration
                continue

            # Calculate the distance between fingertips
            distance_thumb_index = calculate_distance(thumb_tip, index_tip)
            distance_thumb_middle = calculate_distance(thumb_tip, middle_tip)
            distance_thumb_ring = calculate_distance(thumb_tip, ring_tip)
            distance_thumb_pinky = calculate_distance(thumb_tip, pinky_tip)

            # Define a threshold distance for a "closed hand" gesture (when fingers are near each other)
            closed_hand_threshold = 0.03  # You may need to tweak this value depending on your setup

            # Detect if hand is closed (fingers close to each other)
            if (distance_thumb_index < closed_hand_threshold and
                distance_thumb_middle < closed_hand_threshold and
                distance_thumb_ring < closed_hand_threshold and
                distance_thumb_pinky < closed_hand_threshold):
                
                if is_playing:
                    mixer.music.stop()
                    is_playing = False
                    print("Music Stopped!")

            # If the hand is open (fingers are extended), play the music
            else:
                if not is_playing:
                    mixer.music.play()
                    is_playing = True
                    print("Music Playing!")

    # Display the frame
    cv2.imshow("Hand Gesture Music Player", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()