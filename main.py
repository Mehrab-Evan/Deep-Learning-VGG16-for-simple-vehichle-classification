import os
import cv2
import numpy as np

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

base_path = 'car_images'
images, labels, class_names = load_dataset(base_path)


import tensorflow
from keras.models import Sequential
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
# from tensorflow.keras.utils import to_categorical
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Normalize images
images = images / 255.0
labels = to_categorical(labels)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(class_names), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))


def classify_image(image_path, model, class_names):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)
    return class_names[predicted_class]

# Test with a new image
image_path = 'test_bike.jpg'
predicted_class = classify_image(image_path, model, class_names)
print(f'The image is predicted to be: {predicted_class}')


# Save the model
model.save('car_classifier_model.h5')

# Load the model
from keras.models import load_model
# from tensorflow.keras.models import load_model
model = load_model('car_classifier_model.h5')
