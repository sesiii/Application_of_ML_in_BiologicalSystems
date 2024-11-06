import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# Set the data directories
data_dir = 'Dataset2/FNA'
benign_dir = os.path.join(data_dir, 'benign')
malignant_dir = os.path.join(data_dir, 'malignant')
test_dir = 'Dataset2/test'

# Load the image data
benign_images = [os.path.join(benign_dir, f) for f in os.listdir(benign_dir) if f.endswith('.jpg') or f.endswith('.png')]
malignant_images = [os.path.join(malignant_dir, f) for f in os.listdir(malignant_dir) if f.endswith('.jpg') or f.endswith('.png')]
test_images = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.jpg') or f.endswith('.png')]

# Preprocess the images
img_size = (64, 64)
X_benign = np.array([np.array(Image.open(img).resize(img_size)) for img in benign_images])
X_malignant = np.array([np.array(Image.open(img).resize(img_size)) for img in malignant_images])
X_test = np.array([np.array(Image.open(img).resize(img_size)) for img in test_images])

y_benign = np.zeros(len(X_benign))
y_malignant = np.ones(len(X_malignant))
X = np.concatenate((X_benign, X_malignant), axis=0)
y = np.concatenate((y_benign, y_malignant), axis=0)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model with GPU-enabled TensorFlow
with tf.device('/gpu:0'):
    model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val))

# Evaluate the model
train_loss, train_acc = model.evaluate(X_train, y_train)
val_loss, val_acc = model.evaluate(X_val, y_val)
print(f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.4f}')
print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')

# Predict on the test set
y_pred = model.predict(X_test)
y_pred_classes = (y_pred > 0.5).astype(int)

# Save the predictions
np.savetxt('test_predictions.txt', y_pred_classes, fmt='%d')
print(f'Saved test predictions to test_predictions.txt')