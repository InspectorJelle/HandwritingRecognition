# First Data Science / Machine Learning Project

This is my very first project in Data Science and Machine Learning.

I started with a standard CNN implementation, since I do not yet have the knowledge to build one completely from scratch.
To learn and adapt, I added detailed comments to the code in order to understand what every line is doing.

---

## Questions I had while analyzing the code

### Why do I have to normalize the data?
Most images use pixel values in the range **0–255**.
Neural networks train faster and more reliably when the inputs are scaled to a smaller range, typically **0–1** (or sometimes **–1–1**).
This makes gradient descent more stable and less prone to numerical issues, which speeds up training.

---

### Why use CNNs (Convolutional Neural Networks)?
CNNs apply **filters (kernels)** that slide across the image and detect patterns such as edges, textures, or shapes.
Compared to a fully connected network, CNNs require **fewer parameters**, since they reuse the same filter across the whole image.
This makes them both **efficient** and **well-suited for image recognition tasks**.

---

### What does `Conv2D` do and why is the output tensor shrinking?
`Conv2D` applies a 2D convolution: a filter (e.g. `3×3`) slides over the input image.
At each step, the filter weights are multiplied with the image patch, and the results are summed to form a single value in the **feature map**.

Without padding, the output shrinks because the filter cannot extend beyond the image border.
For example:

Input size: 28 × 28
Filter size: 3 × 3
Stride: 1
Output size = 28 – 3 + 1 = 26

So the output tensor becomes **26 × 26**.

---

### Why do I have to expand the array by one dimension?
CNNs expect inputs in the format:

(batch, height, width, channels)

- For grayscale images: channels = 1
- For RGB images: channels = 3

The MNIST dataset only provides `(batch, height, width)` (no channel dimension).
Therefore, I had to add one channel at the end, resulting in shape `(60000, 28, 28, 1)`.

---

### What does Dropout do?
Dropout randomly sets some neurons to **zero during training**.
This forces the model to rely on multiple pathways for learning, which reduces overfitting and improves generalization.

---

## Key Definitions

- **verbose** – Controls how much information is printed during training (e.g., progress bar, loss, accuracy).
- **precision** – The proportion of correctly predicted positives among all predicted positives. *(How reliable are my positive predictions?)*
- **recall** – The proportion of correctly predicted positives among all actual positives. *(How many of the real positives did I find?)*
- **f1-score** – The harmonic mean of precision and recall. Useful when both need to be balanced.
- **optimizer** – The algorithm that updates the model’s weights to minimize the loss function (e.g., Adam, SGD).
- **loss** – A measure of the error between the model’s prediction and the true label, which the optimizer tries to minimize.

---
