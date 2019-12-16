import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, Conv1D, GlobalAveragePooling1D, MaxPooling1D

x_train = np.random.random((100, 64, 100))
y_train = keras.utils.to_categorical(
    np.random.randint(10, size=(100, 1)), num_classes=10)
x_test = np.random.random((20, 64, 100))
y_test = keras.utils.to_categorical(
    np.random.randint(10, size=(20, 1)), num_classes=10)

seq_length = 64

model = Sequential()
model.add(Conv1D(64, 3, activation='relu', input_shape=(seq_length, 100)))
model.add(Conv1D(64, 3, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(128, 3, activation='relu'))
model.add(Conv1D(128, 3, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.5))
model.add(Dense(10, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=6, epochs=10)
score = model.evaluate(x_test, y_test, batch_size=16, verbose=2)
