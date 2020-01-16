from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print(train_images.shape)
print(train_images.ndim)
print(train_images.dtype)

digit = train_images[4]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
