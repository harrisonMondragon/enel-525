from tensorflow.keras import layers, models, preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import os

# Set the path to your dataset
dataset_path = 'smaller_flowers'

# Get the list of class names (assuming each subdirectory is a class)
class_names = sorted(os.listdir(dataset_path))

# Create a dictionary to store file paths for each class
class_files = {class_name: [] for class_name in class_names}

# Populate the dictionary with file paths
for class_name in class_names:
    class_path = os.path.join(dataset_path, class_name)
    class_files[class_name] = [os.path.join(class_path, file) for file in os.listdir(class_path)]

# Split the dataset into training, validation, and test sets
train_files, test_files = train_test_split(list(class_files.values()), test_size=0.3, random_state=42)
valid_files, test_files = train_test_split(test_files, test_size=0.5, random_state=42)

# Define the image dimensions
img_width, img_height = 320, 240
input_shape = (img_width, img_height, 3)  # 3 channels for RGB images

# Define the batch size and number of epochs
batch_size = 32
epochs = 10

# Create a CNN model
model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(len(class_names), activation='softmax'))  # Adjusted for the number of classes

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Data preprocessing using ImageDataGenerator with resizing
datagen = ImageDataGenerator(rescale=1./255)
# datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

# Create generators for training, validation, and test sets
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

test_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    # subset='test'
)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.samples // batch_size)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
