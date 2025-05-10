## 🧠 ThreeLayerConvNet

A from-scratch implementation of a three-layer convolutional neural network (CNN) in NumPy with custom forward and backward passes for convolution, pooling, activation, affine layers, and softmax loss.

## 🔧 Architecture

    Input → Conv → ReLU → MaxPool(2x2) → Affine → ReLU → Affine → Softmax
---

This minimal CNN supports:

Custom weight initialization

Naive forward/backward implementations (no deep learning libraries used)

Layer composability through modular design

Debugging-friendly shape printing for each major block

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
| `3 layer Conv net.ipynb` | Notebook demonstrating usage and testing of the network                |
| `Images`                 | Test Images for out Convolution Layers                                 |  


🙋‍♂️ Credits
Built with by [Yuganter Pratap](https://www.linkedin.com/in/yuganter-pratap-a3a719254/) — inspired by CS231n and DIY deep learning educational projects.
