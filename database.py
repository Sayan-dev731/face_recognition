import cv2
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.preprocessing import LabelEncoder

# Function to capture and save images from video feed
def capture_and_save_images(num_images, save_dir, unique_id):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cap = cv2.VideoCapture(0)

    # Find the maximum index of the last image for the current unique ID
    max_index = 0
    for filename in os.listdir(save_dir):
        if filename.startswith(unique_id + '_'):
            index = int(filename.split('_')[1].split('.')[0])
            max_index = max(max_index, index)

    for i in range(num_images):
        index = max_index + i + 1
        print(f"Capturing image {index} for ID: {unique_id}")

        # Capture image from webcam
        ret, frame = cap.read()

        # Convert image to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Preprocess image (resize, normalize, etc.)
        processed_image = cv2.resize(gray, (128, 128)) / 255.0

        # Save image with unique ID as filename
        image_path = os.path.join(save_dir, f"{unique_id}_{index}.png")
        cv2.imwrite(image_path, processed_image * 255.0)

    cap.release()

# Function to load captured images and labels
def load_images_and_labels(directory):
    images = []
    labels = []

    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            image_path = os.path.join(directory, filename)
            label = filename.split("_")[0]
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            images.append(image)
            labels.append(label)
    
    return np.array(images), np.array(labels)

# Define directory to save captured images
save_dir = "captured_images"

# Capture and save images
num_images_per_person = 300  # Number of images to capture for each person
num_people = 2  # Number of people to capture images for

for i in range(num_people):
    unique_id = input(f"Enter unique ID for person {i+1}: ")
    capture_and_save_images(num_images_per_person, save_dir, unique_id)

# Load captured images and labels
X, y = load_images_and_labels(save_dir)

# Preprocess images
X = X / 255.0  # Normalize pixel values

# Label encoding for the target labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Define and compile model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(len(np.unique(y)), activation='softmax')  # Number of output classes
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=10)

# Save the trained model
model.save('face_recognition_model.h5')
