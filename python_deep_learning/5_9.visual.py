from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras import models
import numpy as np
import matplotlib.pyplot as plt

model = load_model('cats_and_dogs_small_1.h5')
# model.summary()

img_path = 'D:\\dogs-vs-cats\\dogs-vs-cats-small\\test\\cats\\cat.1700.jpg'

img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255
# print(img_tensor.shape)

# plt.imshow(img_tensor[0])
# plt.show()

layer_outputs = [
    layer.output for layer in model.layers[:8]
]
activation_model = models.Model(
    inputs=model.input,
    outputs=layer_outputs
)

activations = activation_model.predict(img_tensor)

first_layer_activation = activations[0]
# print(first_layer_activation.shape)

# plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')
# plt.matshow(first_layer_activation[0, :, :, 7], cmap='viridis')
# plt.show()

# 将每个中间激活的所有通道可视化
layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)

image_per_row = 16

for layer_name, layer_activation in zip(layer_names, activations):
    n_features = layer_activation.shape[-1]

    size = layer_activation.shape[1]

    n_cols = n_features // image_per_row
    display_grid = np.zeros((size * n_cols, image_per_row * size))

    for col in range(n_cols):
        for row in range(image_per_row):
            channel_image = layer_activation[0,
                                             :, :, col * image_per_row + row]
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[
                col * size: (col + 1) * size,
                row * size: (row + 1) * size
            ] = channel_image

    scale = 1. / size
    plt.figure(
        figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0])
    )
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')

plt.show()
