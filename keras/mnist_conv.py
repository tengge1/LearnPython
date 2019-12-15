import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Flatten, Dense, Dropout, Conv2D

(x_train, y_train), (x_test, y_test) = mnist.load_data()

(x_train, x_test) = (x_train/255, x_test/255)

x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

model = Sequential([
    Conv2D(32, (3, 3), input_shape=(28, 28, 1)),
    Flatten(),
    # Dense(128, activation='relu'),
    # Dropout(0.2),
    Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test, y_test, verbose=2)
