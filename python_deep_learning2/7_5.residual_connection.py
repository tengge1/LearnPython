from tensorflow.keras import Input, Model, layers

# 相同

x = Input(shape=(512, 512, 128))
y = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
y = layers.Conv2D(128, 3, activation='relu', padding='same')(y)
y = layers.Conv2D(128, 3, activation='relu', padding='same')(y)

y = layers.add([y, x])

model = Model(x, y)
model.summary()

# 不同

x = Input(shape=(512, 512, 3))
y = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
y = layers.Conv2D(128, 3, activation='relu', padding='same')(y)
y = layers.MaxPooling2D(2, strides=2)(y)

# 将输入形状改为与y相同
residual = layers.Conv2D(128, 1, strides=2, padding='same')(x)

y = layers.add([y, residual])

model = Model(x, y)

model.summary()
