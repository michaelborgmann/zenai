import numpy as np
import keras.ops as kops

x = np.random.random((32,))
y = np.random.random((32,))

z = np.matmul(x, y)  # NumPy
z = x @ y  # Shorthand operator

z = kops.matmul(x, y)  # Keras

print(z)