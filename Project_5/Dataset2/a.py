import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Set the data directories
base_dir = os.path.abspath('Project_5/Dataset2/Dataset2')
benign_dir = os.path.join(base_dir, 'FNA', 'benign')
malignant_dir = os.path.join(base_dir, 'FNA', 'malignant')
test_dir = os.path.join(base_dir, 'test')

# Check if directories exist
if not os.path.exists(benign_dir):
    raise FileNotFoundError(f"The directory {benign_dir} does not exist.")
if not os.path.exists(malignant_dir):
    raise FileNotFoundError(f"The directory {malignant_dir} does not exist.")
if not os.path.exists(test_dir):
    raise FileNotFoundError(f"The directory {test_dir} does not exist.")

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

# Normalize the data
X = X / 255.0
X_test = X_test / 255.0

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

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Check for GPU availability
if tf.config.list_physical_devices('GPU'):
    print("GPU is being used for training.")
else:
    print("GPU is not available. Using CPU for training.")

# Train the model and save the training loss, accuracy, and epochs
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

# Save the training history
with open('output.txt', 'w', encoding='utf-8') as f:
    for epoch in range(20):
        train_loss = history.history['loss'][epoch]
        train_acc = history.history['accuracy'][epoch]
        val_loss = history.history['val_loss'][epoch]
        val_acc = history.history['val_accuracy'][epoch]
        f.write(f'Epoch {epoch + 1}, Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}\n')
        print(f'Epoch {epoch + 1}, Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')

# Plot training & validation loss and accuracy
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.savefig('training_validation_loss_accuracy.png')
plt.show()