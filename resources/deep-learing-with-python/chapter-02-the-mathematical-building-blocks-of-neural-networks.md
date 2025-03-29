# Chapter 2: The Mathematical Building Blocks of Neural Networks

To understand deep learning, it's crucial to grasp some foundational mathematical concepts, such as **tensors**, **tensor operations**, **differentiation**, and **gradient descent**. This chapter aims to build intuition around these ideas while avoiding overly technical details. Instead of using complex mathematical notation, the focus is on clear, executable code, which provides an accurate and unambiguous representation of mathematical operations.

The chapter begins with a practical example of a neural network to introduce the concepts of tensors and gradient descent. Each new concept will be explained in detail, with a focus on clarity and practical understanding. By the end of the chapter, readers should have a solid intuitive understanding of the mathematical foundations of deep learning, preparing them for hands-on work with modern deep learning frameworks like Keras and TensorFlow, which will be covered in subsequent chapters.

## A First Look at a Neural Network

This section introduces a simple neural network built with Keras to classify handwritten digits from the MNIST dataset. MNIST, a widely used benchmark in deep learning, consists of 60,000 training images and 10,000 test images of grayscale digits (28×28 pixels), labeled from 0 to 9.

### Understanding the Dataset

The dataset is preloaded in Keras and consists of NumPy arrays:

```
from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
```

The images are stored as arrays of pixel values, and the labels correspond to digit categories.

Look at the training data:

```
>>> train_images.shape
(60000, 28, 28)
>>> len(train_labels)
60000
>>> train_labels
array([5, 0, 4, ..., 5, 6, 8], dtype=uint8)
```

And the test data:

```
>>> test_images.shape
(10000, 28, 28)
>>> len(test_labels)
10000
>>> test_labels
array([7, 2, 1, ..., 4, 5, 6], dtype=uint8)
```

### Building a Simple Neural Network

A basic feedforward neural network is created using Keras:

```
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(512, activation="relu"),
    layers.Dense(10, activation="softmax")
])
```

Neural networks are built from layers that act as data filters, extracting meaningful representations. Deep learning involves chaining layers to refine data progressively. The described model has two **Dense** (fully connected) layers, with the final layer being a **10-way softmax classifier** that outputs probabilities for digit classification.

* The first **Dense** layer (512 neurons, ReLU activation) extracts meaningful representations.
* The second **Dense** layer (10 neurons, softmax activation) outputs probabilities for each digit class.

**Compiling the Model**

The model is compiled with an optimizer, loss function, and evaluation metric:

```
model.compile(
    optimizer="adam", 
    loss="sparse_categorical_crossentropy", 
    metrics=["accuracy"])
```

* **Adam optimizer**: Adjusts weights efficiently.
* **Sparse categorical crossentropy**: Suitable for multi-class classification.
* **Accuracy**: Monitors classification performance.

### Data Preprocessing

Before training, images are reshaped and scaled to improve learning:

```
train_images = train_images.reshape((60000, 28 * 28)).astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28)).astype("float32") / 255
```

### Training the Model

The model is trained for five epochs using batch gradient descent:

```
model.fit(train_images, train_labels, epochs=5, batch_size=128)
```

During training, the loss function and accuracy are monitored. The model quickly achieves high accuracy (~98.9%) on the training set.

### Evaluating and Making Predictions

Once the model is trained, we can use it to predict class probabilities for unseen data, such as test images:

```
>>> test_digits = test_images[0:10]
>>> predictions = model.predict(test_digits)
>>> predictions[0]
array([1.0726176e-10, 1.6918376e-10, 6.1314843e-08, 8.4106023e-06,
       2.9967067e-11, 3.0331331e-09, 8.3651971e-14, 9.9999106e-01,
       2.6657624e-08, 3.8127661e-07], dtype=float32)
```

The array represents the probability that the test image belongs to each class. In this case, the highest probability is at index 7, suggesting that the digit is a 7:

```
>>> predictions[0].argmax()
7
>>> predictions[0][7]
0.99999106
```

We can check this against the actual test label:

```
>>> test_labels[0]
7
```

To evaluate the model's performance on the unseen test set, we compute the average accuracy:

```
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc}")
```

The test accuracy is 97.8%, which is slightly lower than the 98.9% accuracy on the training data. This gap indicates that the model is overfitting, as it performs better on training data than on new data.

This example shows how to quickly build and train a neural network to classify handwritten digits with just a few lines of Python code. The upcoming chapters will explore the technical details behind tensors, tensor operations, and gradient descent.

### Key Takeaways

* Neural networks process data through **layers** that extract meaningful patterns.
* **Preprocessing** (reshaping and scaling) improves training efficiency.
* **Training** involves optimizing weights using loss functions and gradient descent.
* **Overfitting** occurs when the model performs better on training data than on unseen data.

This example demonstrates how to build, train, and evaluate a neural network in just a few lines of Python. The following chapters delve deeper into the underlying concepts, such as tensors, operations, and gradient descent.

## 2.2 Data Representations for Neural Networks

Tensors are the core data structure in modern machine learning, including frameworks like TensorFlow. They are multidimensional arrays that generalize matrices to an arbitrary number of dimensions, where each dimension is called an axis. Essentially, tensors serve as flexible containers for numerical data, making them fundamental to neural networks and deep learning computations.

### Scalars (rank-0 tensors)

A scalar (rank-0 tensor or 0D tensor) is a tensor containing a single number. In NumPy, this is typically represented as a float32 or float64. You can check its number of axes with the ndim attribute, which will be 0 for scalars.

```
import numpy as np
x = np.array(12)
print(x)       # Output: 12
print(x.ndim)  # Output: 0
```

### Vectors (rank-1 tensors)

A vector (rank-1 tensor or 1D tensor) is an array of numbers with one axis. For example, in NumPy:

```
import numpy as np
x = np.array([12, 3, 6, 14, 7])
print(x)       # Output: [12, 3, 6, 14, 7]
print(x.ndim)  # Output: 1
```

This 5-entry vector has one axis. A common source of confusion is the difference between a "5D vector" (a vector with 5 entries) and a "5D tensor" (a tensor with 5 axes). For clarity, it’s better to refer to the vector as having 5 dimensions along its single axis.

### Matrices (rank-2 tensors)

A matrix (rank-2 tensor or 2D tensor) is an array of vectors with two axes: rows and columns. In NumPy:

```
import numpy as np
x = np.array([[5, 78, 2, 34, 0],
              [6, 79, 3, 35, 1],
              [7, 80, 4, 36, 2]])
print(x.ndim)  # Output: 2
```

In this matrix, the first axis represents the rows, and the second axis represents the columns. For example, [5, 78, 2, 34, 0] is the first row, and [5, 6, 7] is the first column.

### Rank-3 and higher-rank tensors

A rank-3 tensor (or 3D tensor) is created by packing matrices into an array, visually interpretable as a cube of numbers. Example in NumPy:

```
import numpy as np
x = np.array([[[5, 78, 2, 34, 0],
               [6, 79, 3, 35, 1],
               [7, 80, 4, 36, 2]],
              [[5, 78, 2, 34, 0],
               [6, 79, 3, 35, 1],
               [7, 80, 4, 36, 2]],
              [[5, 78, 2, 34, 0],
               [6, 79, 3, 35, 1],
               [7, 80, 4, 36, 2]]])
print(x.ndim)  # Output: 3
```

By stacking rank-3 tensors, you can create rank-4 tensors, and so on. In deep learning, tensors of ranks 0 to 4 are most common, with rank-5 used for video data.

### Key attributes

A tensor has three key attributes:

1. **Number of axes** (rank): Defines the number of axes in a tensor (e.g., a rank-3 tensor has 3 axes).
2. **Shape**: A tuple indicating the number of dimensions along each axis (e.g., shape `(3, 3, 5)` for a rank-3 tensor). A scalar has shape `()`, and a vector has shape `(5,)`.
3. **Data type (dtype)**: The type of data in the tensor (e.g., `float32`, `uint8`, `bool`, etc.).

Example with the MNIST dataset:

```
from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print(train_images.ndim)  # Output: 3
print(train_images.shape)  # Output: (60000, 28, 28)
print(train_images.dtype)  # Output: uint8
```

The `train_images tensor is a rank-3 tensor of 60,000 28×28 grayscale images, stored as 8-bit integers. You can visualize a specific image like so:

```
import matplotlib.pyplot as plt
digit = train_images[4]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
```

The corresponding label for this image is `9:

```
print(train_labels[4])  # Output: 9
```

### Manipulating tensors in NumPy

Tensor slicing allows you to select specific elements or subarrays from a tensor. In NumPy, you can slice tensors by specifying ranges for each axis.

* Example: Select digits #10 to #100 (excluding #100):

```
my_slice = train_images[10:100]
print(my_slice.shape)  # Output: (90, 28, 28)
```

* Equivalent slicing with full axis selection:

```
my_slice = train_images[10:100, :, :]
print(my_slice.shape)  # Output: (90, 28, 28)
```

You can also select specific parts of an image. For example, to extract the bottom-right corner (14x14 pixels):

```
my_slice = train_images[:, 14:, 14:]
```

* Negative indices allow you to slice relative to the end of each axis:

```
my_slice = train_images[:, 7:-7, 7:-7]  # Center 14x14 pixel crop
```

### The notion of data batches

In deep learning, the first axis (axis 0) in data tensors represents the "samples" axis, where each sample could be an image or data point. For instance, in the MNIST example, samples are images of digits.

Deep learning models process data in small batches instead of the entire dataset at once. Here's how to create batches of size 128 from the MNIST dataset:

* First batch:

```
batch = train_images[:128]
```

* Next batch:

```
batch = train_images[128:256]
```

* nth batch:

```
n = 3
batch = train_images[128 * n:128 * (n + 1)]
```

The first axis (axis 0) is often referred to as the batch axis or batch dimension in deep learning frameworks like Keras.

### Real-world examples of data tensors

Data in deep learning typically falls into one of the following tensor categories:

* **Vector data**: Rank-2 tensors of shape (samples, features), where each sample is a vector of numerical attributes (features).
* **Timeseries or sequence data**: Rank-3 tensors of shape (samples, timesteps, features), where each sample is a sequence of feature vectors over time (timesteps).
* **Images**: Rank-4 tensors of shape (samples, height, width, channels), where each sample is a 2D grid of pixels, and each pixel has multiple values (channels).
* **Video**: Rank-5 tensors of shape (samples, frames, height, width, channels), where each sample is a sequence of images (frames).

These categories represent the typical structure of data you will encounter when working with deep learning models.

#### Vector Data

Vector data is a common structure where each data point is represented as a vector, forming a rank-2 tensor. The first axis corresponds to samples, and the second axis corresponds to features.

Examples:

* **Actuarial dataset**: For 100,000 people, each represented by a vector of 3 values (age, gender, income), the dataset is stored in a rank-2 tensor of shape (100000, 3).
* **Text dataset**: For 500 documents, each encoded as a vector of 20,000 values (word counts from a dictionary), the dataset is stored in a tensor of shape (500, 20000).

These examples demonstrate how datasets are structured as rank-2 tensors with samples and features.

#### Timeseries Data or Sequence Data

When data involves time or sequence order, it is typically stored in a rank-3 tensor, with the second axis representing time (or sequence). Each sample is a sequence of vectors, and the dataset is a batch of such sequences.

Examples:

* **Stock price dataset**: Each minute, store the current, highest, and lowest price in a 3D vector. A full trading day (390 minutes) is stored as a matrix of shape (390, 3), and 250 days of data form a rank-3 tensor of shape (250, 390, 3).
* **Tweet dataset**: Each tweet, represented by a sequence of 280 characters from an alphabet of 128 characters, is encoded as a rank-2 tensor (280, 128). A dataset of 1 million tweets forms a tensor of shape (1000000, 280, 128).

#### Image Data

Images typically have three dimensions: height, width, and color depth. Grayscale images, like the MNIST digits, have a single color channel, but are conventionally stored in rank-3 tensors with one channel. A batch of 128 grayscale images (256 × 256) would be stored in a tensor of shape (128, 256, 256, 1), and a batch of color images would be in a tensor of shape (128, 256, 256, 3).

There are two conventions for image tensor shapes:

* **Channels-last** (standard in TensorFlow and JAX): (samples, height, width, color_depth).
* **Channels-first** (standard in PyTorch): (samples, color_depth, height, width).

The Keras API supports both formats.

#### Video Data

Video data requires rank-5 tensors, as a video is a sequence of frames, each being a color image. Each frame can be represented as a rank-3 tensor (height, width, color_depth), and a sequence of frames forms a rank-4 tensor (frames, height, width, color_depth). A batch of videos is represented by a rank-5 tensor of shape (samples, frames, height, width, color_depth).

For example, a 60-second YouTube video (144 × 256 pixels) sampled at 4 frames per second would have 240 frames. A batch of 4 such video clips would have a tensor shape of (4, 240, 144, 256, 3), totaling 106,168,320 values. If stored in `float32`, the tensor size would be 405 MB. Real-world video data is often compressed significantly (e.g., MPEG format) and lighter in storage.

## 2.3 The Gears of Neural Networks: Tensor Operations

Transformations in deep neural networks can be broken down into basic tensor operations, like addition and multiplication, applied to numeric tensors.

For example, in Keras, a Dense layer is defined as:

```
keras.layers.Dense(512, activation="relu")
```

This layer can be viewed as a function that takes a matrix as input and returns another matrix (a transformed version of the input tensor). The specific function is:

```
output = relu(matmul(input, W) + b)
```

Here, there are three key tensor operations:

1. A **tensor product** (matmul) between the input tensor and a weight matrix `W`.
2. An **addition** (+) of the resulting matrix with a bias vector `b`.
3. A **ReLU** operation, which returns the maximum of `x` and 0 (`relu(x) = max(x, 0)`).

NOTE:  `relu` stands for "REctified Linear Unit"

**Tip**: While the section focuses on linear algebra, mathematical concepts are expressed through Python code (using NumPy and TensorFlow), which may be easier for non-mathematicians to understand.

## 2.4 Element-wise operations

Element-wise operations (e.g., `relu`, addition) are applied independently to each entry in tensors. These operations are ideal for parallel execution. A naive Python implementation using loops is much slower compared to using optimized libraries like **NumPy**.

**Naive implementation** in Python:

```
def naive_relu(x):
    assert len(x.shape) == 2
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] = max(x[i, j], 0)
    return x
```

Using **NumPy** for the same operations:

```
import numpy as np
z = x + y
z = np.maximum(z, 0.)
```

The NumPy version is much faster, utilizing **BLAS** (Basic Linear Algebra Subprograms) for efficient tensor manipulation.

When running JAX, TensorFlow, or PyTorch on a GPU, elementwise operations are performed using fully-vectorized CUDA implementations, which optimize the GPU's highly-parallel architecture.

## 2.5 Broadcasting

Broadcasting allows element-wise operations on tensors of different shapes by expanding the smaller tensor to match the larger one. This involves:

1. Adding axes to the smaller tensor to match the larger tensor's dimensions.
2. Repeating the smaller tensor along the new axes.

For example, adding a matrix `X` with shape (32, 10) and a vector `y` with shape (10,):

```
import numpy as np
X = np.random.random((32, 10))
y = np.random.random((10,))
y = np.expand_dims(y, axis=0)  # shape becomes (1, 10)
Y = np.concatenate([y] * 32, axis=0)  # shape becomes (32, 10)
```

The operation is computationally efficient, as the repetition of `y` is virtual, not creating new memory allocations.

A naive implementation for adding a matrix and vector is:

```
def naive_add_matrix_and_vector(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert x.shape[1] == y.shape[0]
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += y[j]
    return x
```

Broadcasting can also be used for operations like the element-wise maximum:

```
x = np.random.random((64, 3, 32, 10))
y = np.random.random((32, 10))
z = np.maximum(x, y)
```

## 2.6 Tensor Product

The **tensor product** (or **dot product**, **matrix multiplication**) is a fundamental operation in machine learning. It computes **scalar**, **vector**, **matrix**, or **higher-dimensional tensor outputs**, depending on input dimensions.

**NumPy & Keras implementations:**

```
import numpy as np
import keras.ops as kops

x = np.random.random((32,))
y = np.random.random((32,))

z = np.matmul(x, y)  # NumPy
z = x @ y  # Shorthand operator
z = kops.matmul(x, y)  # Keras
```

**Mathematically**: z=x⋅y

### 1. Vector Dot Product

The **dot product of two vectors** produces a **scalar** by summing element-wise products:

```
def naive_vector_product(x, y):
    assert len(x.shape) == 1 and len(y.shape) == 1
    assert x.shape[0] == y.shape[0]
    z = 0.
    for i in range(x.shape[0]):
        z += x[i] * y[i]
    return z
```

✅ Only works for vectors of the same length.

### 2. Matrix-Vector Product

Multiplying a **matrix** x (shape **(m, n)**) by a **vector** y
y (shape **(n,)**) results in a **vector** (shape **(m,)**):

```
def naive_matrix_vector_product(x, y):
    assert len(x.shape) == 2 and len(y.shape) == 1
    assert x.shape[1] == y.shape[0]
    z = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            z[i] += x[i, j] * y[j]
    return z
```

Or, using the **previous dot product function** for efficiency:

```
def naive_matrix_vector_product(x, y):
    z = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        z[i] = naive_vector_product(x[i, :], y)
    return z
```

✅ Each row of the matrix is multiplied by the vector.

### 3. Matrix-Matrix Product

Multiplying **two matrices** x (shape **(m, n)**) and y (shape **(n, p)**) results in a **matrix** (shape **(m, p)**), with each element computed as a **dot product between rows of** 
**x and columns of y**:

```
def naive_matrix_product(x, y):
    assert len(x.shape) == 2 and len(y.shape) == 2
    assert x.shape[1] == y.shape[0]  # Ensure shape compatibility
    z = np.zeros((x.shape[0], y.shape[1]))
    for i in range(x.shape[0]):
        for j in range(y.shape[1]):
            row_x = x[i, :]
            column_y = y[:, j]
            z[i, j] = naive_vector_product(row_x, column_y)
    return z
```

✅ The result has shape (rows of x, columns of y).

### 4. Generalized Tensor Products

Matrix multiplication extends to higher-dimensional tensors using the same shape compatibility rules:

```
(a, b, c, d) • (d,) -> (a, b, c)  # Vector product  
(a, b, c, d) • (d, e) -> (a, b, c, e)  # Matrix product  
```

✅ Works for tensors with arbitrary axes, maintaining dimensional consistency.

## 2.7 Tensor Reshaping

**Tensor reshaping** is a key operation that reorganizes the structure of data without changing the total number of elements. It is essential for preparing inputs in deep learning models.

1. **Reshaping Basics**

Reshaping modifies the **dimensions** of a tensor but **preserves the total number of elements**. Example:

```
import numpy as np

x = np.array([[0., 1.],
              [2., 3.],
              [4., 5.]])
print(x.shape)  # (3, 2)

x = x.reshape((6, 1))  # Reshape to (6,1)
print(x)
# Output:
# [[0.]
#  [1.]
#  [2.]
#  [3.]
#  [4.]
#  [5.]]

x = x.reshape((2, 3))  # Reshape to (2,3)
print(x)
# Output:
# [[0. 1. 2.]
#  [3. 4. 5.]]
```

✅ Shape can change, but the number of elements remains constant.

2. **Transposing a Tensor**

**Transposition** swaps rows and columns, effectively flipping a matrix:

```
x = np.zeros((300, 20))
x = np.transpose(x)
print(x.shape)
```

✅ Converts shape (300, 20) → (20, 300).
✅ Equivalent to x.T in NumPy.

###  2.7.1 Geometric interpretation of tensor operations

Tensor operations can be interpreted geometrically, as tensor values represent coordinates in a geometric space.

**Vector Addition as Translation**

A vector can be seen as an arrow from the origin to a point in space. Adding two vectors results in shifting the first vector by the second. For example:

```
A = [0.5, 1]
B = [1, 0.25]
C = A + B  # Vector sum
```

Geometrically, this operation translates **A** by **B**. When applied to an entire object (a set of points), it results in **translation**.

**Common Geometric Transformations as Tensor Operations**

* **Translation**: Adding a vector shifts points without distortion.
* **Rotation**: A counterclockwise rotation of a 2D vector by an angle **θ** can be achieved with matrix multiplication:
```
R = [[cos(θ), -sin(θ)],
     [sin(θ),  cos(θ)]]
```
* **Scaling**: Resizes an object by multiplying with a **diagonal matrix**:
```
S = [[sx,  0],
     [ 0, sy]]
```
* **Linear Transform**: Any matrix multiplication `W @ x applies a **linear transformation**, including scaling and rotation.
* **Affine Transform**: A combination of linear transformation and translation:
```
y = W @ x + b
```

This is exactly what a **Dense** layer computes.

**Dense Layer with ReLU Activation**

An affine transformation remains affine even when applied multiple times:

```
affine2(affine1(x)) = W2 @ (W1 @ x + b1) + b2
                     = (W2 @ W1) @ x + (W2 @ b1 + b2)
```

Without activation functions, stacking **Dense** layers is equivalent to a single **Dense** layer, making the model a **linear transformation**. Activation functions like **ReLU** introduce non-linearity, allowing **Deep Neural Networks** to learn complex transformations.

This concept is expanded further in the next chapter.

## 2.8 A Geometric Interpretation of Deep Learning

Neural networks perform **geometric transformations** on data through **sequential tensor operations**, forming a complex mapping in high-dimensional space.

**The Paper Ball Analogy**

Imagine two **stacked sheets of paper** (red & blue) **crumpled into a ball**, where each sheet represents a data class. A neural network's goal is to **unfold** this ball, making the classes **separable**.

**Deep Learning as Stepwise Unfolding**

* Input data lies on a **folded manifold** (continuous surface).
* **Machine learning** reshapes this manifold to make data easier to classify.
* **Deep networks** break this process into **incremental transformations**.
* Each **layer** slightly **untangles** the data.
* A **deep stack of layers** makes even highly complex **data separable**.

Deep learning succeeds by **progressively simplifying** complex data structures, making them **linearly separable** through layers of transformations.

## 2.9 The Engine of Neural Networks: Gradient-Based Optimization

Neural networks use **trainable parameters** (weights and biases) to transform input data through layers, typically expressed as:

```
output = relu(matmul(input, W) + b)
```

* **W** and **b** are the weight matrices and bias terms.
* Initially, these weights are randomly initialized and don't produce useful outputs. The next step is to **adjust the weights** based on feedback, which is the **training process**.

**Training Loop:**

1. Draw a batch of training samples (x) and corresponding targets (y_true).
2. Run a **forward pass** to obtain predictions (y_pred).
3. Compute the **loss**, measuring the mismatch between predictions (y_pred) and true targets (y_true).
4. **Update the weights** to reduce the loss.

This loop continues until the loss is sufficiently low, meaning the model learns to map inputs to correct targets.

**Weight Update (Gradient Descent):**

* A naive approach involves manually adjusting each weight by testing different values, which is inefficient. Instead, we use **gradient descent**:
	* Functions like `matmul` and `+` are **differentiable**, meaning small changes in input result in small, predictable changes in output.
	* The **gradient** describes how the loss changes with respect to the model's weights.
	* By computing the gradient, we can adjust all the weights in a single update step, moving them in the direction that minimizes the loss.

This method drastically improves efficiency and is the core optimization technique behind modern neural networks.

### 2.9.1 What’s a Derivative?

A derivative represents how a function changes as its input changes. For a continuous, smooth function `f(x) = y`, a small change in `x` results in a small change in `y`, denoted as `ε_y.

For a small change `ε_x` in `x`, we can approximate the change in `y` as:

```
f(x + ε_x) = y + a * ε_x
```

Where **a** is the **slope** at point `p`, called the **derivative** of `f` at `p`. The derivative tells you:

* If `a` is negative, increasing `x` decreases `f(x)`.
* If `a` is positive, increasing `x` increases `f(x).
* The **magnitude** of `a` indicates how quickly the change occurs.

For any differentiable function, there exists a **derivative function** `f'(x)` that maps input values `x` to the slope of `f` at those points. For example:

* The derivative of `cos(x)` is `-sin(x)`.
* The derivative of `f(x) = a * x is f'(x) = a`.

**Why Derivatives Matter:**

* Derivatives are key for **optimization**: they help identify how to adjust `x` to minimize `f(x).
* To minimize `f(x)`, you move `x` in the opposite direction of the derivative.

### 2.9.2 Derivative of a Tensor Operation: The Gradient

In the case of a scalar function, the derivative gives the slope of a curve. For tensor functions (functions that take inputs such as vectors, matrices, or higher-dimensional tensors), the derivative is generalized as the **gradient**. The gradient describes how the output of a function changes with respect to its input tensors.

**Example:**

* **Input**:
	* `x` (input vector)
	* `W` (weights matrix of the model)
	* `y_true` (target output)
	* `loss` (loss function)

The function computes:
```
y_pred = matmul(W, x)
loss_value = loss(y_pred, y_true)
```

The goal is to update `W` to minimize `loss_value`.

* **Gradient**:
	The gradient `grad(loss_value, W0)` represents the rate of change in the loss with respect to each coefficient in `W` at a particular point `W0`. Specifically, it describes the curvature of `loss_value = f(W)` at `W0`.

The gradient is computed as:
```
grad(loss_value, W0) = [grad_ij(f(W), W[i,j])]
```

**Updating the Weights:**

To reduce the loss, you update `W by moving it **opposite** to the gradient:

```
W1 = W0 - step * grad(f(W0), W0)
```

* `step` is a small scaling factor controlling how much the weights are adjusted.

**Intuition:**

* The gradient represents how the loss surface "curves" in the space of weights `W`.
* Just like adjusting a scalar function based on its derivative, you adjust a tensor by moving against its gradient to minimize the loss.

## 2.10 Stochastic Gradient Descent (SGD)

Finding the minimum of a differentiable function can theoretically be done by solving the equation `grad(f(W), W) = 0` for `W`. However, this approach is computationally infeasible for neural networks, which often have millions or billions of parameters. Instead, **stochastic gradient descent** (SGD) is used, iteratively adjusting parameters based on small updates.

**Mini-Batch Stochastic Gradient Descent**

Instead of computing the gradient on the entire dataset (which is expensive), we update the weights using a randomly sampled batch. The steps are:

1. Draw a batch of training samples `x` and their targets `y_true.
2. Run the model to obtain predictions y_pred (forward pass).
3. Compute the loss between `y_pred` and `y_true`.
4. Compute the gradient of the loss with respect to model parameters (backward pass).
5. Update the parameters in the opposite direction of the gradient:
```
W -= learning_rate * gradient
```
where `learning_rate` controls the step size.

* **True SGD**: Uses a batch of size 1 (single sample).
* **Batch Gradient Descent**: Uses the entire dataset for updates.
* **Mini-Batch SGD** (most common): Uses a small subset of the data per update.

Choosing a good `learning_rate` is crucial:
* **Too small** → Slow convergence.
* **Too large** → The updates may be erratic and fail to reach the minimum.

**Optimization Methods & Momentum**

To improve SGD, optimizers consider previous weight updates. One important variant is **SGD with momentum**, which helps avoid **local minima** and speeds up convergence.

* Think of optimization as a **ball rolling down a hill**:
	* If it has momentum, it won’t get stuck in small dips (local minima).
	* Instead of relying only on the current gradient, it also considers past updates.

**Momentum-Based Update (Simplified Code)**

```
past_velocity = 0.
momentum = 0.1
while loss > 0.01:
    w, loss, gradient = get_current_parameters()
    velocity = past_velocity * momentum - learning_rate * gradient
    w = w + momentum * velocity - learning_rate * gradient
    past_velocity = velocity
    update_parameter(w)
```

* `momentum` controls how much past updates influence the next step.
* This method smooths out updates and prevents getting stuck in bad local minima.

**Other Optimization Methods**

Several advanced optimizers build on SGD:
* **Adagrad**
* **RMSprop**
* **Adam** (Adaptive Moment Estimation)

These methods dynamically adjust the learning rate and improve convergence speed.

## 2.11 Chaining Derivatives: The Backpropagation Algorithm

Computing the gradient of complex functions is essential for training neural networks. Since a neural network consists of multiple layers of differentiable operations, we use **backpropagation** to efficiently compute gradients by leveraging the **chain rule** from calculus.

### 2.11.1 The Chain Rule

The chain rule states that if we have two functions `f` and `g`, their composition `fg(x) = f(g(x)) has a derivative:

```
grad(y, x) = grad(y, x1) * grad(x1, x)
```
where `x1 = g(x)` and `y = f(x1)`.

For multiple nested functions:
```
def fghj(x):
    x1 = j(x)
    x2 = h(x1)
    x3 = g(x2)
    y = f(x3)
    return y

grad(y, x) = grad(y, x3) * grad(x3, x2) * grad(x2, x1) * grad(x1, x)
```

Backpropagation applies this rule to compute gradients in **deep neural networks**, where layers are composed of differentiable operations like matrix multiplication, activation functions (`relu`, `softmax`), and loss functions.

#### Applying the Chain Rule in Neural Networks

For a two-layer network, the loss function can be written as:
```
loss_value = loss(y_true, softmax(matmul(relu(matmul(inputs, W1) + b1), W2) + b2))
```

Each operation (such as `matmul`, `relu`, `softmax`) has a known derivative. Using the **chain rule**, we can compute the gradient of the loss with respect to each parameter (`W1`, `b1`, `W2`, `b2`).

**This forms the foundation of backpropagation**, an efficient algorithm that computes gradients layer by layer, propagating the error backward from the loss to the weights.

### 2.11.2 Automatic Differentiation with Computation Graphs

Backpropagation can be understood using **computation graphs**, a fundamental data structure in deep learning frameworks like TensorFlow and PyTorch. A computation graph represents tensor operations as a **directed acyclic graph** (DAG), allowing efficient differentiation and optimization.

#### Computation Graphs and Their Benefits

A computation graph represents **computations as data**, enabling:

* **Automatic differentiation**: The framework computes derivatives without manual coding.
* **Optimization and parallelism**: Computations can be distributed across multiple devices.
* **Transformation and analysis**: Graphs can be optimized or modified programmatically.

For example, the computation graph of a simple neural network layer:

```
loss_value = loss(y_true, softmax(matmul(relu(matmul(inputs, W1) + b1), W2) + b2))
```

Each operation (`matmul`, `relu`, `softmax`, `loss`) has a known derivative, forming a structured computation graph that can be **differentiated automatically**.

#### Forward and Backward Pass

Consider a simple model with scalars:

* Input `x`, target `y_true`, weights `w`, `b`
* Forward computation:

```
x1 = x * w  
x2 = x1 + b  
loss_val = abs(y_true - x2)
```

Given `x=2`, `y_true=4`, `w=3`, `b=1`, forward pass gives `loss_val = abs(4 - (2*3 + 1)) = 1`.

* **Backward pass**: We reverse the graph and compute gradients using the **chain rule**.

```
grad(loss_val, x2) = 1
grad(x2, x1) = 1
grad(x2, b) = 1
grad(x1, w) = 2
```

Applying the chain rule:

```
grad(loss_val, w) = 1 * 1 * 2 = 2
grad(loss_val, b) = 1 * 1 = 1
```

* If multiple paths exist in the backward graph, gradients are summed.

#### Automatic Differentiation in Modern Frameworks

Backpropagation is implemented using automatic differentiation in frameworks like:

* **TensorFlow** (`tf.GradientTape`)
* **PyTorch** (`torch.autograd`)
* **JAX** (`jax.grad`)

These tools handle differentiation **automatically**, so you don’t need to implement backpropagation manually—unlike early deep learning implementations in C!

## 2.12 Reviewing Our First Neural Network Example

Now that we've built a solid understanding of the mathematical foundations of neural networks, let’s revisit the first example and analyze it with this deeper knowledge. Initially, the neural network seemed like a "black box," but now we can break it down into clear components:

1. **Input data processing**
2. **Model architecture**
3. **Loss function and optimization**
4. **Training process**

### 1. Preparing the Input Data

The dataset we used was the MNIST handwritten digits dataset, where each image is a 28×28 grayscale matrix. We reshaped these images into 1D tensors (vectors) of size 784 and normalized the pixel values to a range of 0 to 1 for better numerical stability:

```
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255
```

Now, we understand that:

* The images are stored in NumPy tensors (`float32`), reshaped into vectors of length 784.
* Normalization (dividing by 255) ensures stable weight updates during training.

### 2. Model Architecture

We defined a simple feedforward neural network using Keras:

```
model = keras.Sequential([
    layers.Dense(512, activation="relu"),
    layers.Dense(10, activation="softmax")
])
```

Breaking it down:

* **Two `Dense layers**: The first has 512 neurons with ReLU activation, and the second has 10 neurons with softmax activation for classification.
* **Weight tensors**: Each layer contains trainable weight matrices that transform input data. These are adjusted through training to improve predictions.

### 3. Model Compilation (Loss and Optimizer)

```
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
```

What we now understand:

* **Loss function**: `"sparse_categorical_crossentropy"` measures how far predictions are from true labels, guiding the learning process.
* **Optimizer** (`adam`): A variant of gradient descent that adapts learning rates dynamically.
* **Minimizing loss**: The model updates weights iteratively to reduce this loss function.

### 4. Training Process

```
model.fit(train_images, train_labels, epochs=5, batch_size=128)
```

Key insights:

* **Mini-batch gradient descent**: The model processes batches of 128 images at a time.
* **Epochs**: The dataset is processed 5 times in total.
* **Backpropagation**: Uses the chain rule to compute gradients and update weights.
* **Total updates**: 469 mini-batches per epoch, totaling 2,345 updates.

After training, the model achieves **high accuracy** on digit classification.

### 2.12.1 Reimplementing our first example from scratch

To deepen our understanding, we now reimplement the neural network with explicit computation steps, avoiding high-level Keras functions. While we won’t rewrite tensor operations or backpropagation, this exercise clarifies the core mathematical processes behind deep learning. The next chapter will cover the Keras API in detail, but for now, this hands-on approach helps solidify key concepts.

#### A Simple Dense Class

The Dense layer applies the transformation:

```
output = activation(matmul(W, input) + b)
```

where `W` and `b` are trainable parameters. The `NaiveDense` class implements this by:
* Creating `W` (weights) and `b` (bias) as TensorFlow variables.
* Applying matrix multiplication and bias addition.
* Optionally applying an activation function.

```
import keras
from keras import ops

class NaiveDense:
    def __init__(self, input_size, output_size, activation=None):
        self.activation = activation
        self.W = keras.Variable(
            shape=(input_size, output_size), initializer="uniform")
        self.b = keras.Variable(
            shape=(output_size,), initializer="zeros")

    def __call__(self, inputs):
        x = ops.matmul(inputs, self.W) + self.b
        return self.activation(x) if self.activation else x

    @property
    def weights(self):
        return [self.W, self.b]
```

This makes the layer’s computation explicit without using high-level Keras functions.

#### A Simple Sequential Class

The `NaiveSequential` class chains multiple layers together, applying them sequentially to the input. It:
* Stores a list of layers.
* Passes inputs through each layer in order via `__call__()`.
* Provides a weights property to track all layer parameters.

```
class NaiveSequential:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x

    @property
    def weights(self):
        return [w for layer in self.layers for w in layer.weights]
```

Using NaiveDense and `NaiveSequential`, we can define a simple neural network:

```
model = NaiveSequential([
    NaiveDense(input_size=28 * 28, output_size=512, activation=ops.relu),
    NaiveDense(input_size=512, output_size=10, activation=ops.softmax)
])
assert len(model.weights) == 4
```

This mimics a basic Keras model, explicitly handling layer operations and weight management.

#### A Batch Generator

To process the MNIST dataset in mini-batches, we define a BatchGenerator class. It:

* Stores the dataset and batch size.
* Computes the total number of batches.
* Iterates through the dataset, returning a batch of images and labels.

```
import math

class BatchGenerator:
    def __init__(self, images, labels, batch_size=128):
        assert len(images) == len(labels)
        self.index = 0
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.num_batches = math.ceil(len(images) / batch_size)

    def next(self):
        images = self.images[self.index : self.index + self.batch_size]
        labels = self.labels[self.index : self.index + self.batch_size]
        self.index += self.batch_size
        return images, labels
```

This enables efficient mini-batch processing, essential for training neural networks with large datasets.

### 2.12.2 Running one training step

The training step updates model weights after processing a batch of data. It involves:

1. Computing model predictions.
2. Calculating the loss based on true labels.
3. Computing gradients of the loss with respect to weights.
4. Updating the weights in the opposite direction of the gradient.

```
def one_training_step(model, images_batch, labels_batch):
    predictions = model(images_batch)
    loss = ops.sparse_categorical_crossentropy(labels_batch, predictions)
    average_loss = ops.mean(loss)
    gradients = get_gradients_of_loss_wrt_weights(loss, model.weights)
    update_weights(gradients, model.weights)
    return loss
```

#### Weight Update Step

Weights are updated using the computed gradients and a small learning rate:

```
learning_rate = 1e-3

def update_weights(gradients, weights):
    for g, w in zip(gradients, weights):
        w.assign(w - g * learning_rate)
```

Alternatively, Keras provides optimizers to handle weight updates:

```
from keras import optimizers

optimizer = optimizers.SGD(learning_rate=1e-3)

def update_weights(gradients, weights):
    optimizer.apply_gradients(zip(gradients, weights))
```

#### Gradient Computation with TensorFlow

Instead of manually deriving gradients, we use TensorFlow’s `GradientTape` for automatic differentiation:

```
import tensorflow as tf

x = tf.zeros(shape=())
with tf.GradientTape() as tape:
    y = 2 * x + 3
grad_of_y_wrt_x = tape.gradient(y, x)
```

Using `GradientTape`, we rewrite `one_training_step()` without needing a separate gradient function:

```
def one_training_step(model, images_batch, labels_batch):
    with tf.GradientTape() as tape:
        predictions = model(images_batch)
        loss = ops.sparse_categorical_crossentropy(labels_batch, predictions)
        average_loss = ops.mean(loss)
    gradients = tape.gradient(average_loss, model.weights)
    update_weights(gradients, model.weights)
    return average_loss
```

With this in place, we are ready to implement a full training epoch.

## 2.13 The full training loop

The full training loop consists of running multiple epochs, where each epoch involves processing all batches in the training dataset. The training step (`one_training_step()`) is applied to each batch, updating the model’s weights.

```
def fit(model, images, labels, epochs, batch_size=128):
    for epoch_counter in range(epochs):
        print(f"Epoch {epoch_counter}")
        batch_generator = BatchGenerator(images, labels, batch_size)
        for batch_counter in range(batch_generator.num_batches):
            images_batch, labels_batch = batch_generator.next()
            loss = one_training_step(model, images_batch, labels_batch)
            if batch_counter % 100 == 0:
                print(f"loss at batch {batch_counter}: {loss:.2f}")
```

### Testing the Full Loop

To test the loop, the MNIST dataset is loaded, preprocessed, and then passed into the training function:

```
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255

fit(model, train_images, train_labels, epochs=10, batch_size=128)
```

This completes the full training process for the neural network.

## 2.14 Evaluating the model

To evaluate the model, we calculate the accuracy by comparing the model's predictions to the true labels of the test dataset. Specifically, we:

1. Get the model's predictions for the test images.
2. Use `argmax` to select the class with the highest probability for each prediction.
3. Compare the predicted labels with the actual test labels to compute the accuracy.

```
predictions = model(test_images)
predicted_labels = ops.argmax(predictions, axis=1)
matches = predicted_labels == test_labels
print(f"accuracy: {ops.mean(matches):.2f}")
```

Although it’s a lot of work to manually implement these steps, you now have a clear understanding of the processes behind `fit()` in Keras. This knowledge will help you better use Keras’s high-level features while understanding how things work under the hood.
