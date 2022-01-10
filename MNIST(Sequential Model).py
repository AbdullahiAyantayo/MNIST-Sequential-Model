import tensorflow
from tensorflow.keras.layers import Input, MaxPool2D, GlobalAvgPool2D, Flatten, Conv2D
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.python.keras.layers.core import Dense
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.layers.pooling import GlobalMaxPool2D

model = tensorflow.keras.Sequential(
    [
        Input(shape = (28, 28, 1)),
        Conv2D(32, (3, 3), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPool2D(),
        BatchNormalization(),

        Conv2D(64, (3, 3), activation='relu'),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPool2D(),
        BatchNormalization(),

        GlobalMaxPool2D(),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(10, activation='softmax')
    ]
)

(train_image, train_label), (test_image, test_label) = tensorflow.keras.datasets.mnist.load_data()


train_image = train_image.astype('float32') / 255.0
test_image = test_image.astype('float32') / 255.0

train_image = np.expand_dims(train_image, axis=-1)
test_image = np.expand_dims(test_image, axis=-1)

print("train_image.shape = ", train_image.shape)
print("train_label.shape = ", train_label.shape)
print("test_image.shape = ", test_image.shape)
print("test_label.shape = ", test_label.shape)

model.summary()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
model.fit(train_image, train_label, batch_size=64, epochs=3, validation_split=0.15)
model.evaluate(test_image, test_label, batch_size=64)