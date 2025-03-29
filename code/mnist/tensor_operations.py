import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

x = np.array(12)

print(x)
print(x.ndim)

x = np.array([12, 3, 6, 14, 7])
print(x)
print(x.ndim)

x = np.array([[5, 78, 2, 34, 0],
              [6, 79, 3, 35, 1],
              [7, 80, 4, 36, 2]])
print(x.ndim)

x = np.array([[[5, 78, 2, 34, 0],
               [6, 79, 3, 35, 1],
               [7, 80, 4, 36, 2]],
              [[5, 78, 2, 34, 0],
               [6, 79, 3, 35, 1],
               [7, 80, 4, 36, 2]],
              [[5, 78, 2, 34, 0],
               [6, 79, 3, 35, 1],
               [7, 80, 4, 36, 2]]])
print(x.ndim)

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print(train_images.ndim)
print(train_images.shape)
print(train_images.dtype)

digit = train_images[4]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()

print(train_labels[4])

my_slice = train_images[10:100]
print(my_slice.shape)

my_slice = train_images[10:100, :, :]
print(my_slice.shape)

my_slice = train_images[10:100, 0:28, 0:28]
print(my_slice.shape)

my_slice = train_images[:, 14:, 14:]
print(my_slice.shape)

my_slice = train_images[:, 7:-7, 7:-7]
print(my_slice.shape)

batch = train_images[:128]
print(batch.shape)

batch = train_images[128:256]
print(batch.shape)

n = 3
batch = train_images[128 * n:128 * (n + 1)]
print(batch.shape)