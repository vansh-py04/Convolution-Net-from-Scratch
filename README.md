## 🧠 ThreeLayerConvNet

A from-scratch implementation of a three-layer convolutional neural network (CNN) in NumPy with custom forward and backward passes for convolution, pooling, activation, affine layers, and softmax loss.

## 🔧 Architecture

    Input → Conv → ReLU → MaxPool(2x2) → Affine → ReLU → Affine → Softmax
---

This minimal CNN supports:

* Custom weight initialization

* Naive forward/backward implementations (no deep learning libraries used)

* Layer composability through modular design

* Debugging-friendly shape printing for each major block

---

## 📁 Files Overview
| File                     | Description                                                            |
| ------------------------ | ---------------------------------------------------------------------- |
| `CNN.py`                 | Defines the `ThreeLayerConvNet` class with full forward/backward logic |
| `Conv2d.py`              | Naive 2D convolution forward and backward pass                         |
| `ReLU.py`                | ReLU activation forward and backward pass                              |
| `Maxpool.py`             | Naive 2x2 max pooling forward and backward                             |
| `Affine_layer.py`        | Fully-connected (affine) layer logic                                   |
| `Softmax_loss.py`        | Softmax loss and gradient implementation                               |
| `Sandwich_layers.py`     | Helper functions for chaining layers: `conv-relu-pool`, `affine-relu`  |
| `Implementation.ipynb` | Notebook demonstrating usage and testing of the network                |
| `Images`                 | Test Images for out Convolution Layers                                 |  

The model can classify images of shape (3, 32, 32) into 10 classes. You can also plug it into datasets like CIFAR-10 (after appropriate preprocessing).

---

## 📌 Features

NumPy-only: No TensorFlow/PyTorch/Keras used

Backpropagation from scratch: Every layer implements its own gradient logic

Debug output: Key print statements show internal shape flows

Extensible design: Each component is modular and reusable

This repository is designed not just to use a CNN — but to help you learn how it actually works under the hood. Every major layer of a convolutional neural network has been manually implemented from scratch using just NumPy, with readable and modular Python code.

Here’s how you can use this repo to gain a deep understanding of CNN fundamentals:

# Study Each Layer in Isolation
Each layer (e.g., convolution, ReLU, max-pooling, affine, softmax) is implemented in its own file:

* Conv2d.py: Learn how convolutions apply filters using nested loops and stride/padding logic.

* ReLU.py: Understand how non-linear activations shape model capacity.

* Maxpool.py: Observe how spatial downsampling is performed via max-pooling.

* Affine_layer.py: Explore fully-connected transformations using matrix reshaping and dot products.

* Softmax_loss.py: Learn how classification loss and gradients are computed for training.

## Clone the Repo
    git clone https://github.com/vansh-py04/Convolution-Net-from-Scratch.git
    cd ThreeLayerConvNet

🙋‍♂️ Credits
Built with by [Yuganter Pratap](https://www.linkedin.com/in/yuganter-pratap-a3a719254/) — inspired by CS231n and DIY deep learning educational projects.
