# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Input

# Generate input data with specific shape
num_samples = 30
x_train = np.random.randint(255, size=(num_samples, 28, 28, 3))
y_train = np.random.randint(2, size=num_samples) # 0, 1

# ---------------------- Functional API ----------------------
# Specify the input shape
height = 28
width = 28
ch = 3 #RGB

# Create an input node (batchsize is specified at runtime)
inputnode = tensorflow.keras.Input(shape=(height,width,ch)) #[None,h,w,c]

# Create intermediate layers manually
conv_layer = Conv2D(10,(3,3), activation='relu')
conv1 = conv_layer(inputnode)
pool1 = MaxPool2D()(conv1)
flatten = Flatten()(pool1)
dense1 = Dense(10, activation='relu')(flatten) # Fully connected layer

# Create an output node/view model summary and train the model
outputnode = Dense(1, activation='sigmoid')(dense1)
model = tensorflow.keras.Model(inputs=inputnode, outputs=outputnode)
model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(x_train, y_train, epochs=100)

### Optional: Viewing the feature Map (How the CNN Understands Images)

# ### The sequential model ###
# model = Sequential()
# model.add(Conv2D(10,(3,3), activation='relu'))
# model.add(MaxPool2D())
# model.add(Flatten())
# model.add(Dense(10,activation='relu'))
# model.add(Dense(1,activation='sigmoid'))
# model.compile(optimizer='adam', loss='binary_crossentropy')
