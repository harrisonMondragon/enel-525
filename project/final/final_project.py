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





# Make predictions on the test set
predictions = model.predict(test_generator)

# Get the class indices for each class name
class_indices = {class_name: i for i, class_name in enumerate(class_names)}

# Display the first 10 images from the test set along with true and predicted labels
num_images_to_display = 10

for i in range(num_images_to_display):
    # Get the true class label
    true_label_index = np.argmax(test_labels[i])
    true_label = class_names[true_label_index]

    # Get the predicted class label
    predicted_label_index = np.argmax(predictions[i])
    predicted_label = class_names[predicted_label_index]

    # Load and display the image
    img_path = os.path.join(dataset_path, test_images[i])
    img = preprocessing.image.load_img(img_path, target_size=(img_height, img_width))
    img_array = preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the pixel values to be between 0 and 1

    # Display the image
    plt.imshow(img_array[0])
    plt.title(f'True: {true_label}, Predicted: {predicted_label}')
    plt.show()





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
