from tensorflow.keras import Model, Input
from tensorflow.keras import layers

x = Input(shape=(512, 512, 3))
brantch_a = layers.Conv2D(128, 1, activation='relu', strides=2)(x)
brantch_b = layers.Conv2D(128, 1, activation='relu')(x)
brantch_b = layers.Conv2D(128, 3, activation='relu', strides=2)(brantch_b)

branch_c = layers.AveragePooling2D(3, strides=2)(x)
branch_c = layers.Conv2D(128, 3, activation='relu')(branch_c)

branch_d = layers.Conv2D(128, 1, activation='relu')(x)
branch_d = layers.Conv2D(128, 3, activation='relu')(branch_d)
branch_d = layers.Conv2D(128, 3, activation='relu', strides=2)(branch_d)

output = layers.concatenate(
    [brantch_a, brantch_b, branch_c, branch_d], axis=-1)

model = Model(x, output)

model.summary()
