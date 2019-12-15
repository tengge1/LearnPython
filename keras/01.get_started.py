import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import categorical_crossentropy

model = Sequential()

model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=10, activation='softmax'))

model.compile(
    loss=categorical_crossentropy,
    optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True)
)
