import os
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.models import load_model
import matplotlib.pyplot as plt

# Function to load images from a folder and assign a label
def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.resize(img, (128, 128))  # Resize to a consistent size
            images.append(img)
            labels.append(label)
    return images, labels

# Function to load the entire dataset
def load_dataset(base_path):
    images = []
    labels = []
    class_names = os.listdir(base_path)
    for idx, class_name in enumerate(class_names):
        class_path = os.path.join(base_path, class_name)
        class_images, class_labels = load_images_from_folder(class_path, idx)
        images.extend(class_images)
        labels.extend(class_labels)
    return np.array(images), np.array(labels), class_names

# Base path for the dataset
base_path = 'car_images'
images, labels, class_names = load_dataset(base_path)

# Normalize images
images = images / 255.0
labels = to_categorical(labels)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
datagen.fit(X_train)

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(class_names), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with data augmentation
history = model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=50, validation_data=(X_test, y_test))

# Save the model
model.save('car_classifier_model.h5')

# Load the model
model = load_model('car_classifier_model.h5')

# Function to classify a single image
def classify_image(image_path, model, class_names):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)
    return predicted_class

# Function to display images side by side
def display_images(image_path1, image_path2):
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)
    img1 = cv2.resize(img1, (1500, 1500))
    img2 = cv2.resize(img2, (1500, 1500))

    # Convert BGR images to RGB for displaying with matplotlib
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # Display images
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title("Input Image")
    plt.imshow(img1)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Matched Class Image")
    plt.imshow(img2)
    plt.axis('off')

    plt.show()

# Test with a new image
image_path = 'test_bus2.PNG'
predicted_class = classify_image(image_path, model, class_names)
print(f'The image is predicted to be: {class_names[predicted_class]}')

# Find a sample image from the predicted class
predicted_class_folder = os.path.join(base_path, class_names[predicted_class])
sample_image_path = os.path.join(predicted_class_folder, os.listdir(predicted_class_folder)[0])

# Display the input image and the matched class image
display_images(image_path, sample_image_path)

# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()
