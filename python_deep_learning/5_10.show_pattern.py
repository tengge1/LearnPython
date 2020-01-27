import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

# tensorflow 2.0 禁用eager模式
tf.compat.v1.disable_eager_execution()

model = VGG16(weights='imagenet', include_top=False)


def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    x += 0.5
    x = np.clip(x, 0, 1)

    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def generate_pattern(layer_name, filter_index, size=150):
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])

    grads = K.gradients(loss, model.input)[0]

    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    iterate = K.function([model.input], [loss, grads])

    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128

    step = 1
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

        img = input_img_data[0]
        return deprocess_image(img)


plt.imshow(generate_pattern('block3_conv1', 0))
plt.show()
