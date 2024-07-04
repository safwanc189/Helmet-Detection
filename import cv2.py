import cv2
import numpy as np

# Load pre-trained Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load pre-trained Haar Cascade for helmet detection (you would need to train this)
helmet_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_motorcycle_helmet.xml')

# Load a pre-trained machine learning model (SVM, Random Forest, etc.)
# This should be a model trained on a helmet dataset (not provided here)

# Open a video capture
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Loop through detected faces
    for (x, y, w, h) in faces:
        # Extract the region of interest (ROI) for helmet detection
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Detect helmets in the ROI
        helmets = helmet_cascade.detectMultiScale(roi_gray)

        # Loop through detected helmets and draw bounding boxes
        for (hx, hy, hw, hh) in helmets:
            cv2.rectangle(roi_color, (hx, hy), (hx + hw, hy + hh), (255, 0, 0), 2)

    # Display the frame with bounding boxes
    cv2.imshow('Helmet Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()