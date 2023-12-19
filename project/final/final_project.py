from tensorflow.keras import layers, models, preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# Set the path to dataset, please use an absolute path
dataset_path = 'C:\\Users\\Harry\\Desktop\\ENEL525\\enel-525\\project\\final\\flowers'

# Define the image dimensions
img_width = 320
img_height = 240
input_shape = (img_width, img_height, 3)  # 3 channels for RGB images

# Define the batch size and number of epochs
batch_size = 32
epochs = 10

# Create a list of all image paths and labels
image_paths = []
labels = []
class_names = []

for class_folder in os.listdir(dataset_path):
    class_names.append(class_folder)
    class_folder_path = os.path.join(dataset_path, class_folder)

    if os.path.isdir(class_folder_path):
        # Construct image paths for the class
        class_image_paths = [os.path.join(class_folder_path, image) for image in os.listdir(class_folder_path)]

        image_paths.extend(class_image_paths)
        labels.extend([class_folder] * len(class_image_paths))

# Convert lists to numpy arrays
image_paths = np.array(image_paths)
labels = np.array(labels)

# Shuffle the data
indices = np.arange(len(image_paths))
np.random.shuffle(indices)
image_paths = image_paths[indices]
labels = labels[indices]

# Manually split the data into training, validation, and test sets
split_index_train = int(len(image_paths) * 0.7)
split_index_val = split_index_train + int(len(image_paths) * 0.15)

train_images = image_paths[:split_index_train]
train_labels = labels[:split_index_train]

val_images = image_paths[split_index_train:split_index_val]
val_labels = labels[split_index_train:split_index_val]

test_images = image_paths[split_index_val:]
test_labels = labels[split_index_val:]

# Use ImageDataGenerator to create data generators with normalization for each set
datagen = ImageDataGenerator(rescale=1./255)

# Create data generators for each set
train_generator = datagen.flow_from_dataframe(
    dataframe=pd.DataFrame({'filename': train_images, 'class': train_labels}),
    directory=dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

validation_generator = datagen.flow_from_dataframe(
    dataframe=pd.DataFrame({'filename': val_images, 'class': val_labels}),
    directory=dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

test_generator = datagen.flow_from_dataframe(
    dataframe=pd.DataFrame({'filename': test_images, 'class': test_labels}),
    directory=dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

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
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')

# Plot training loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot training accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Load the first 10 images from the test set
num_images_to_display = 10

# Map class indices to class names
class_indices = test_generator.class_indices
class_names = list(class_indices.keys())

# Lists to store information for all 10 predictions
true_labels = []
predicted_labels = []
images_to_display = []

for i in range(num_images_to_display):
    # Load and preprocess the image with the correct target size
    img_path = os.path.join(dataset_path, test_generator.filenames[i])
    img = preprocessing.image.load_img(img_path, target_size=(img_width, img_height))
    img_array = preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Make a prediction
    prediction = model.predict(img_array)

    # Decode predictions
    true_label = class_names[test_generator.classes[i]]
    predicted_label = class_names[np.argmax(prediction)]

    # Collect information for display
    true_labels.append(true_label)
    predicted_labels.append(predicted_label)
    images_to_display.append(img)

# Plot all 10 images and their predictions on the same pyplot window
plt.figure(figsize=(10, 6))
for i in range(num_images_to_display):
    plt.subplot(2, 5, i + 1)
    plt.imshow(images_to_display[i])
    plt.title(f'True: {true_labels[i]}\nPredicted: {predicted_labels[i]}')
    plt.axis('off')

plt.tight_layout()
plt.show()
