from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
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

plt.imshow(img_tensor[0])
plt.show()
