import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def compute_confusion_matrix(true, pred):
    K = len(np.unique(true)) # Number of classes
    result = np.zeros((K, K))
    for i in range(len(true)):
        result[true[i]][pred[i]] += 1
    return result

# Reading data from csv files
dataset = pd.read_csv("FourClasses.csv")
targets = pd.read_csv("Targets_0.csv")

# Converting datasets to numpy arrays
dataset = dataset.to_numpy()
targets = targets.to_numpy()

# Get rid of 'indices' take class 0 and class 1 only for Exercise 1 (2000 data points)
dataset = dataset[0:2000,1:] # Taking column index 1 and 2 only
targets = targets[0:2000,1] # Taking column index 1 only

# Plotting the dataset in a scatter plot
plt.scatter(dataset[:,0], dataset[:,1], c=targets)
plt.title("Two-class dataset")
plt.xlabel("Height (cm)")
plt.ylabel("Weight (kg)")
plt.show()

# Since out dataset is not high in magnitude, standardization is optional, but it is good practice to do so

# Split dataset into train and test subsets
np.random.seed(seed=0) # Set a seed for repeatability
permuted_idx = np.random.permutation(dataset.shape[0]) # Permute sequence of the dataset
x_train = dataset[permuted_idx[0:1600]]
y_train = targets[permuted_idx[0:1600]]
x_test = dataset[permuted_idx[1600:]]
y_test = targets[permuted_idx[1600:]]

print(f"Number of datapoints in x_train: {len(x_train)}")
print(f"Number of datapoints in x_test: {len(x_test)}")
print(f"Dataset shape: {dataset.shape}")

# Build, compile, and train neural network
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(50, activation='sigmoid'))
model.add(tf.keras.layers.Dense(50, activation='sigmoid'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid')) # Binary classification sigmoid
model.compile(optimizer='adam', loss='binary_crossentropy')
history = model.fit(x_train, y_train, batch_size=24, epochs=100)

model.summary() # View model summary
y_pred = model.predict(x_test) # Test the model using test data
y_pred_t = 1 * (y_pred > 0.5) # Computing predicted test labels/targets (Threshold at 0.5)
y_pred_t = y_pred_t.astype(int) # Converting to integer

# Compute and view the confusion matrix
confusion_mx = compute_confusion_matrix(y_test, y_pred_t)
print(confusion_mx)

# Compute accuracy
Accuracy = ((confusion_mx[0,0] + confusion_mx[1,1]) / len(y_test)) * 100
print(f"Accuracy: {Accuracy} %")
