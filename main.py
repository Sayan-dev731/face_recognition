import cv2
import numpy as np
from tensorflow.keras.models import load_model # type: ignore

# Load saved model
model = load_model('face_recognition_model.h5')

# Load face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Dictionary to map numerical labels to unique IDs
label_to_unique_id = {0: 'sayan', 1: 'naman'}  # Update with actual unique IDs

# Function to recognize faces and display unique IDs
def recognize_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Recognize faces
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        resized_roi = cv2.resize(face_roi, (128, 128)) / 255.0
        resized_roi = np.expand_dims(resized_roi, axis=-1)
        prediction = model.predict(np.array([resized_roi]))
        predicted_label = np.argmax(prediction)
        confidence = np.max(prediction)
        
        # Draw green rectangle around the face
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display unique ID
        unique_id = label_to_unique_id.get(predicted_label, 'Unknown')
        cv2.putText(image, f'ID: {unique_id} ({confidence:.2f})', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return image

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Recognize faces and display unique IDs
    frame = recognize_faces(frame)

    # Display the frame
    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()


